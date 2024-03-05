# adapted from https://github.com/jik876/hifi-gan/blob/master/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from hifigan2.utils import get_padding, load_generator_from_checkpoint
import math


URLS = {
    "hifigan": "https://github.com/bshall/hifigan/releases/download/v0.1/hifigan-67926ec6.pt",
    "hifigan-hubert-soft": "https://github.com/bshall/hifigan/releases/download/v0.1/hifigan-hubert-discrete-bbad3043.pt",
    "hifigan-hubert-discrete": "https://github.com/bshall/hifigan/releases/download/v0.1/hifigan-hubert-soft-65f03469.pt",
}

LRELU_SLOPE = 0.1


class HifiganGenerator(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        upsample_kernel_sizes: Tuple[int, ...] = (20, 8, 4, 4),
        upsample_initial_channel: int = 512,
        #upsample_factors: int = (10, 4, 2, 2),
        # upsample_factors: int = (10, 4, 2, 2),
        upsample_factors: int = (8, 4, 4, 2),
        inference_padding: int = 5,
        sample_rate: int = 16000,
    ) -> None:
        r"""HiFiGAN Generator
        Args:
            in_channels (int): number of input channels.
            resblock_dilation_sizes (Tuple[Tuple[int, ...], ...]): list of dilation values in each layer of a `ResBlock`.
            resblock_kernel_sizes (Tuple[int, ...]): list of kernel sizes for each `ResBlock`.
            upsample_kernel_sizes (Tuple[int, ...]): list of kernel sizes for each transposed convolution.
            upsample_initial_channel (int): number of channels for the first upsampling layer. This is divided by 2
                for each consecutive upsampling layer.
            upsample_factors (Tuple[int, ...]): upsampling factors (stride) for each upsampling layer.
            inference_padding (int): constant padding applied to the input at inference time.
            sample_rate (int): sample rate of the generated audio.
        """
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.sample_rate = sample_rate
        self.in_channels = in_channels
        # initial upsampling layers
        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        )

        # upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock1(ch, k, d))
        # post convolution layer
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv_pre(x)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    @torch.no_grad()
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.inference_padding, self.inference_padding), "replicate")
        return self(x), self.sample_rate


    @torch.no_grad()
    def generateMe(self, x: torch.Tensor, segment_length: int, hop_length: int, verbose:bool= False) -> torch.Tensor:
        if verbose: print("####HIFIGAN GENERATOR x before ", x.shape)
        assert x.size(0)==1 and len(x.shape)==3 and x.size(1)==self.in_channels, f"Shape mismatch {x.shape}"
        mel_length_size=2
        mel_frames_per_segment = math.ceil(segment_length / hop_length)
        if verbose: print("####HIFIGAN GENERATOR mel_frames_per_segment ",mel_frames_per_segment," self.in_channels ",self.in_channels)
        x = F.pad(x, (0,  math.ceil(x.size(mel_length_size)/mel_frames_per_segment)*mel_frames_per_segment - x.size(mel_length_size)), "constant", 0)
        if verbose: print("####HIFIGAN GENERATOR x after pad ", x.shape)

        x2 = x.transpose(2,1).reshape(-1,mel_frames_per_segment, self.in_channels).transpose(2,1)
        if verbose: print("####HIFIGAN GENERATOR x after view ", x.shape," x2 ",x2.shape)

        #x = x.view(-1, x.size(1), mel_frames_per_segment)
        # mel_diff = 0#src_logmel.size(-1) - mel_frames_per_segment if self.train else 0
        # mel_offset = 0#random.randint(0, max(mel_diff, 0))
        # frame_offset = hop_length * mel_offset
        # if x.size(mel_length_size) >= self.segment_length// hop_length:
        #     x = x[
        #                  :, :, mel_offset: mel_offset + mel_frames_per_segment
        #                  ]
        # elif x.size(mel_length_size) < self.segment_length// hop_length:
        #     #wav = F.pad(wav, (0, self.segment_length - wav.size(-1)))
        #     x = F.pad(
        #         x,
        #         (0, mel_frames_per_segment - x.size(-1)),
        #         "constant",
        #         x.min(),
        #     )
        x2 = F.pad(x2, (self.inference_padding, self.inference_padding), "replicate")
        if verbose: print("####HIFIGAN GENERATOR x2 after pad2 ", x2.shape)
        hf_outp= self(x2)
        if verbose: print("####HIFIGAN GENERATOR hf_outp after generator ", hf_outp.shape)
        hf_outp= hf_outp.view(1, 1, -1)
        if verbose: print("####HIFIGAN GENERATOR hf_outp after generator and view ", hf_outp.shape)
        return hf_outp, self.sample_rate

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class ResBlock1(torch.nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: Tuple[int, ...] = (1, 3, 5)
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: Tuple[int, ...] = (1, 3)
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


def _hifigan(
    name: str,
    pretrained: bool = True,
    progress: bool = True,
    map_location=None,
) -> HifiganGenerator:
    hifigan = HifiganGenerator()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[name], map_location=map_location, progress=progress
        )
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        hifigan.load_state_dict(checkpoint)
        hifigan.eval()
        hifigan.remove_weight_norm()
    return hifigan


def hifigan(
    pretrained: bool = True, progress: bool = True, map_location=None
) -> HifiganGenerator:
    return _hifigan("hifigan", pretrained, progress, map_location)


def hifigan_hubert_soft(
    pretrained: bool = True, progress: bool = True, map_location=None
) -> HifiganGenerator:
    return _hifigan("hifigan-hubert-soft", pretrained, progress, map_location=None)


def hifigan_hubert_discrete(
    pretrained: bool = True, progress: bool = True, map_location=None
) -> HifiganGenerator:
    return _hifigan("hifigan-hubert-discrete", pretrained, progress, map_location=None)


def create_generator_and_load_checkpoint(
    load_path,
    #generator,
    # discriminator,
    # optimizer_generator,
    # optimizer_discriminator,
    # scheduler_generator,
    # scheduler_discriminator,finetune=False,
    rank
    # logger,
    ):
    generator = HifiganGenerator().to(rank)
    #generator = DDP(generator, device_ids=[rank])
    load_generator_from_checkpoint(load_path= load_path, generator=generator, rank=rank )

    # generator = DDP(generator, device_ids=[rank])
    #
    # optimizer_generator = optim.AdamW(
    #     generator.parameters(),
    #     lr=BASE_LEARNING_RATE if not args.finetune else FINETUNE_LEARNING_RATE,
    #     betas=BETAS,
    #     weight_decay=WEIGHT_DECAY,
    # )
    #
    # # scheduler_generator = optim.lr_scheduler.ExponentialLR(
    # #     optimizer_generator, gamma=LEARNING_RATE_DECAY
    # # )
    # # melspectrogram = LogMelSpectrogram().to(rank)

    # logger.info(f"Loading checkpoint from {load_path}")
    #checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    #generator.load_state_dict(checkpoint["generator"]["model"])

    return generator
