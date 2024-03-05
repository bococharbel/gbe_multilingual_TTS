import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt


import scipy.io.wavfile
import os
import random
import subprocess
import numpy as np
from scipy.io.wavfile import read

import librosa

from scipy.io.wavfile import read


def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)



def load_wav_to_torch(full_path, force_sampling_rate=None):
    if os.path.basename(full_path).endswith(".npy"):
        data = np.load(full_path)
        sampling_rate= force_sampling_rate
    #elif force_sampling_rate is not None:
    else:
        try:
            sampling_rate, data = read(full_path)
        except:
            data, sampling_rate = librosa.load(full_path, sr=force_sampling_rate)

    if len(data.shape) == 2:
        data = data[:, 0]

    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data - 128) / 128.0

    #data = data.astype(np.float32)

    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_wav_to_numpy(full_path, force_sampling_rate=None):
    if os.path.basename(full_path).endswith(".npy"):
        data = np.load(full_path)
        sampling_rate= force_sampling_rate
    #elif force_sampling_rate is not None:
    else:
        try:
            sampling_rate, data = read(full_path)
        except:
            data, sampling_rate = librosa.load(full_path, sr=force_sampling_rate)

    if len(data.shape) == 2:
        data = data[:, 0]

    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data - 128) / 128.0

    #data = data.astype(np.float32)
    return data.astype(np.float32), sampling_rate
    #return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_wav_to_torch_hifigan(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    #print("####",f"number of channels = {data.shape[1]}")
    return torch.from_numpy(data).float(), sampling_rate


def get_padding(k, d):
    return int((k * d - d) / 2)


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def save_checkpoint(
    checkpoint_dir,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    step,
    loss,
    best,
    logger,
):
    state = {
        "generator": {
            "model": generator.state_dict(),
            "optimizer": optimizer_generator.state_dict(),
            "scheduler": scheduler_generator.state_dict(),
        },
        "discriminator": {
            "model": discriminator.state_dict(),
            "optimizer": optimizer_discriminator.state_dict(),
            "scheduler": scheduler_discriminator.state_dict(),
        },
        "step": step,
        "loss": loss,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"model-{step}.pt"
    torch.save(state, checkpoint_path)
    if best:
        best_path = checkpoint_dir / "model-best.pt"
        torch.save(state, best_path)
    logger.info(f"Saved checkpoint: {checkpoint_path.stem}")


def load_checkpoint(
    load_path,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    rank,
    logger,
    finetune=False,
):
    logger.info(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    generator.load_state_dict(checkpoint["generator"]["model"])
    discriminator.load_state_dict(checkpoint["discriminator"]["model"])
    if not finetune:
        optimizer_generator.load_state_dict(checkpoint["generator"]["optimizer"])
        scheduler_generator.load_state_dict(checkpoint["generator"]["scheduler"])
        optimizer_discriminator.load_state_dict(
            checkpoint["discriminator"]["optimizer"]
        )
        scheduler_discriminator.load_state_dict(
            checkpoint["discriminator"]["scheduler"]
        )
    return checkpoint["step"], checkpoint["loss"]

def load_generator_from_checkpoint(
    load_path,
    generator,
    # discriminator,
    # optimizer_generator,
    # optimizer_discriminator,
    # scheduler_generator,
    # scheduler_discriminator,finetune=False,
    rank
    # logger,
    ):
    #generator = HifiganGenerator().to(rank)
    #discriminator = HifiganDiscriminator().to(rank)

    # generator = DDP(generator, device_ids=[rank])
    # discriminator = DDP(discriminator, device_ids=[rank])
    #
    # optimizer_generator = optim.AdamW(
    #     generator.parameters(),
    #     lr=BASE_LEARNING_RATE if not args.finetune else FINETUNE_LEARNING_RATE,
    #     betas=BETAS,
    #     weight_decay=WEIGHT_DECAY,
    # )
    # optimizer_discriminator = optim.AdamW(
    #     discriminator.parameters(),
    #     lr=BASE_LEARNING_RATE if not args.finetune else FINETUNE_LEARNING_RATE,
    #     betas=BETAS,
    #     weight_decay=WEIGHT_DECAY,
    # )
    #
    # # scheduler_generator = optim.lr_scheduler.ExponentialLR(
    # #     optimizer_generator, gamma=LEARNING_RATE_DECAY
    # # )
    # # scheduler_discriminator = optim.lr_scheduler.ExponentialLR(
    # #     optimizer_discriminator, gamma=LEARNING_RATE_DECAY
    # # )
    # # melspectrogram = LogMelSpectrogram().to(rank)

    # logger.info(f"Loading checkpoint from {load_path}")
    #state_dict=None
    checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    ckpt_state_dict = checkpoint["generator"]["model"]
    print(ckpt_state_dict.keys())
    #generator.load_state_dict(checkpoint["generator"]["model"])
    model_dict = generator.state_dict()
    for k in ckpt_state_dict.keys():
        if k in model_dict and model_dict[k].shape==ckpt_state_dict[k].shape:
            pname = k
            pval = ckpt_state_dict[k]
            print("load_finetune_checkpoint loading ",k)
            model_dict[pname] = pval.clone().to(model_dict[pname].device)
        elif k.replace("module.","") in model_dict and model_dict[k.replace("module.","")].shape==ckpt_state_dict[k].shape:
            #pname = k
            pval = ckpt_state_dict[k]
            print("load_finetune_checkpoint loading ",k," in ", k.replace("module.",""))
            model_dict[k.replace("module.","")] = pval.clone().to(model_dict[k.replace("module.","")].device)


        #self.load_state_dict(model_dict)
    generator.load_state_dict(model_dict)

    # if args.resume is not None:
    #     global_step, best_loss = load_checkpoint(
    #         load_path=args.resume,
    #         generator=generator,
    #         discriminator=discriminator,
    #         optimizer_generator=optimizer_generator,
    #         optimizer_discriminator=optimizer_discriminator,
    #         scheduler_generator=scheduler_generator,
    #         scheduler_discriminator=scheduler_discriminator,
    #         rank=rank,
    #         logger=logger,
    #         finetune=args.finetune,
    #     )
    # else:
    #     global_step, best_loss = 0, float("inf")
    return generator
