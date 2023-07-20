# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import torch

from common.text.symbols import get_symbols, get_pad_idx
from fastpitch.model import FastPitch, FastPitchMulti, FastPitchMultiGenerated
from fastpitch.modelwav import FastPitchMultiwav
from fastpitch.modelwavvq import FastPitchMultiwavVq
from fastpitch.modelwavvqttsasr import FastPitchMultiwavVqTtsAsr
#from coqui_ai.model import FastPitch, FastPitchMulti, FastPitchMultiGenerated

from coqui_ai.modelaligner import FastPitchMultiWithAligner
from coqui_ai.modelfastspeech2 import FastSpeechModel
from coqui_ai.modelfastspeech2vq import FastSpeechModelwavVq
from coqui_ai.modelfastspeech2vq import FastSpeechModelwavVq as FastSpeechModelwavVqTtsAsr
from coqui_ai.modelmixertts import FastPitchMultiWithMixer
from coqui_ai.modelvits import  FastPitchMultiWithVITSAligner 
from coqui_ai.tacotron2 import Tacotron2

from fastpitch.model_jit import FastPitchJIT
from waveglow.model import WaveGlow
from hifigan.models import Generator as HifiganGenerator


def parse_model_args(model_name, parser, add_help=False):
    if model_name.upper() == 'WAVEGLOW':
        from waveglow.arg_parser import parse_waveglow_args
        return parse_waveglow_args(parser, add_help)
    elif model_name.upper() == 'HIFIGAN' or model_name.upper() == 'HIFI-GAN' or model_name.upper() == 'HI-FI-GAN':
        from hifigan.arg_parser import parse_hifigan_args
        return parse_hifigan_args(parser, add_help)
    elif model_name.upper() == 'FASTPITCH':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name.upper() == 'FASTPITCHMULTI':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name.upper() == 'FASTPITCHMULTIWAV':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name.upper() == 'FASTPITCHMULTIWAVVQ':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name.upper() == 'FASTPITCHMULTIWAVVQTTSASR':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name.upper() == 'FASTPITCHMULTIGENERATED':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)

    elif model_name.upper() == 'FASTPITCHMULTIWITHALIGNER':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name.upper() == 'FASTSPEECHMODEL':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name.upper() == 'FASTSPEECHMODELWAVVQ':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name.upper() == 'FASTSPEECHMODELWAVVQTTSASR':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name.upper() == 'FASTPITCHMULTIWITHMIXER':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name.upper() == 'FASTPITCHMULTIWITHVITSALIGNER':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name.upper() == 'TACOTRON2':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    else:
        raise NotImplementedError(model_name)


def init_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if module.affine:
            module.weight.data.uniform_()
    for child in module.children():
        init_bn(child)


def get_model(model_name, model_config, device,
              uniform_initialize_bn_weight=False, forward_is_infer=False,
              jitable=False):

    if model_name.upper() == 'WAVEGLOW':
        model = WaveGlow(**model_config)

    elif model_name.upper() == 'HIFIGAN' or model_name.upper() == 'HIFI-GAN' or model_name.upper() == 'HI-FI-GAN':
        model = HifiganGenerator(**model_config)

    elif model_name.upper() == 'FASTPITCH':
        if jitable:
            model = FastPitchJIT(**model_config)
        else:
            model = FastPitch(**model_config)

    elif model_name.upper() == 'FASTPITCHMULTI':
        if jitable:
            raise NotImplementedError("JIT"+odel_name)#model = FastPitchJIT(**model_config)
        else:
            model = FastPitchMulti(**model_config)

    elif model_name.upper() == 'FASTPITCHMULTIWAV':
        if jitable:
            raise NotImplementedError("JIT"+odel_name)#model = FastPitchJIT(**model_config)
        else:
            model = FastPitchMultiwav(**model_config)

    elif model_name.upper() == 'FASTPITCHMULTIWAVVQ':
        if jitable:
            raise NotImplementedError("JIT"+odel_name)#model = FastPitchJIT(**model_config)
        else:
            model = FastPitchMultiwavVq(**model_config)

    elif model_name.upper() == 'FASTPITCHMULTIWAVVQTTSASR':
        if jitable:
            raise NotImplementedError("JIT"+odel_name)#model = FastPitchJIT(**model_config)
        else:
            model = FastPitchMultiwavVqTtsAsr(**model_config)

    elif model_name.upper() == 'FASTPITCHMULTIGENERATED':
        if jitable:
            raise NotImplementedError("JIT"+model_name)#model = FastPitchJIT(**model_config)
        else:
            model = FastPitchMultiGenerated(**model_config)
  

    elif model_name.upper() == 'FASTPITCHMULTIWITHALIGNER':
        if jitable:
            raise NotImplementedError("JIT"+model_name)#model = FastPitchJIT(**model_config)
        else:
            model = FastPitchMultiWithAligner(**model_config)


    elif model_name.upper() == 'FASTSPEECHMODEL':
        if jitable:
            raise NotImplementedError("JIT"+model_name)#model = FastPitchJIT(**model_config)
        else:
            model = FastSpeechModel(**model_config)

    elif model_name.upper() == 'FASTSPEECHMODELWAVVQ':
        if jitable:
            raise NotImplementedError("JIT"+model_name)#model = FastPitchJIT(**model_config)
        else:
            model = FastSpeechModelwavVq(**model_config)

    elif model_name.upper() == 'FASTSPEECHMODELWAVVQTTSASR':
        if jitable:
            raise NotImplementedError("JIT"+model_name)#model = FastPitchJIT(**model_config)
        else:
            model = FastSpeechModelwavVqTtsAsr(**model_config)

    elif model_name.upper() == 'FASTPITCHMULTIWITHMIXER':
        if jitable:
            raise NotImplementedError("JIT"+model_name)#model = FastPitchJIT(**model_config)
        else:
            model = FastPitchMultiWithMixer(**model_config)


    elif model_name.upper() == 'FASTPITCHMULTIWITHVITSALIGNER':
        if jitable:
            raise NotImplementedError("JIT"+model_name)#model = FastPitchJIT(**model_config)
        else:
            model = FastPitchMultiWithVITSAligner(**model_config)


    elif model_name.upper() == 'TACOTRON2':
        if jitable:
            raise NotImplementedError("JIT"+model_name)#model = FastPitchJIT(**model_config)
        else:
            model = Tacotron2(**model_config)

    else:
        raise NotImplementedError(model_name)

    if forward_is_infer:
        model.forward = model.infer

    if uniform_initialize_bn_weight:
        init_bn(model)

    return model.to(device)


def get_model_config(model_name, args):
    """ Code chooses a model based on name"""
    if model_name.upper() == 'WAVEGLOW':
        model_config = dict(
            n_mel_channels=args.n_mel_channels,
            n_flows=args.flows,
            n_group=args.groups,
            n_early_every=args.early_every,
            n_early_size=args.early_size,
            WN_config=dict(
                n_layers=args.wn_layers,
                kernel_size=args.wn_kernel_size,
                n_channels=args.wn_channels
            )
        )
        return model_config
    elif  model_name.upper() == 'HIFIGAN' or model_name.upper() == 'HIFI-GAN' or model_name.upper() == 'HI-FI-GAN':
        model_config = dict(
            resblock_kernel_sizes=args.resblock_kernel_sizes,
            upsample_rates=args.upsample_rates,
            upsample_kernel_sizes=args.upsample_kernel_sizes,
            upsample_initial_channel=args.upsample_initial_channel,
            resblock=args.resblock, resblock_dilation_sizes=args.resblock_dilation_sizes,
        )
        #print(model_config) #, **model_config
        return model_config
    elif model_name.upper() == 'FASTPITCH':
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=len(get_symbols(args.symbol_set)), #args.n_symbols, #
            padding_idx=get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
        )
    elif model_name.upper() == 'FASTPITCHMULTI':
        #print(args)
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type, 
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim,
            use_glow_mas=args.use_glow_mas,
            use_mas=args.use_mas,
            use_soft_duration_predictor= args.use_soft,
            use_tgt_dur_in_soft=args.use_tgt_dur_soft,
           # use_soft_tacotron_duration_predictor=args.use_soft_tacotron,
        )
        return model_config
    elif model_name.upper() == 'FASTPITCHMULTIWAV':
        #print(args)
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type,
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim,
            use_glow_mas=args.use_glow_mas,
            use_mas=args.use_mas,
            use_soft_duration_predictor= args.use_soft,
            use_tgt_dur_in_soft=args.use_tgt_dur_soft,
            use_soft_tacotron_duration_predictor=args.use_soft_tacotron,
            use_diffusion_decoder= args.use_diffusion, use_prior_grad=args.use_priorgrad,
            use_hifigan_wav_decoder=args.use_hifigan_wav_decoder
        )
        return model_config
    elif model_name.upper() == 'FASTPITCHMULTIWAVVQ':
        #print(args)
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type,
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim,
            use_glow_mas=args.use_glow_mas,
            use_mas=args.use_mas,
            use_soft_duration_predictor= args.use_soft,
            use_tgt_dur_in_soft=args.use_tgt_dur_soft,
            use_soft_tacotron_duration_predictor=args.use_soft_tacotron,
            use_diffusion_decoder= args.use_diffusion, use_prior_grad=args.use_priorgrad,
            use_hifigan_wav_decoder=args.use_hifigan_wav_decoder,
            use_msmc_hifigan_wav_decoder=args.use_msmc_hifigan_wav_decoder,
            use_vq_gan=args.use_vq_gan,
            use_vq_frame_decoder=args.use_vq_frame_decoder,
            vq_pred_mel= args.vq_pred_mel, vq_use_gan_output_for_mel_generation=args.vq_use_gan_output_for_mel_generation,
            vq_use_fastpitch_transformer =args.vq_use_fastpitch_transformer, use_conformer=args.use_conformer,
            can_train_vq_only= args.can_train_vq_only, train_vq_only= args.train_vq_only
        )
        return model_config
    elif model_name.upper() == 'FASTPITCHMULTIWAVVQTTSASR':
        #print(args)
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type,
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim,
            use_glow_mas=args.use_glow_mas,
            use_mas=args.use_mas,
            use_soft_duration_predictor= args.use_soft,
            use_tgt_dur_in_soft=args.use_tgt_dur_soft,
            use_soft_tacotron_duration_predictor=args.use_soft_tacotron,
            use_diffusion_decoder= args.use_diffusion, use_prior_grad=args.use_priorgrad,
            use_hifigan_wav_decoder=args.use_hifigan_wav_decoder,
            use_msmc_hifigan_wav_decoder=args.use_msmc_hifigan_wav_decoder,
            use_vq_gan=args.use_vq_gan,
            use_vq_frame_decoder=args.use_vq_frame_decoder,
            vq_pred_mel= args.vq_pred_mel, vq_use_gan_output_for_mel_generation=args.vq_use_gan_output_for_mel_generation,
            vq_use_fastpitch_transformer =args.vq_use_fastpitch_transformer, use_conformer=args.use_conformer,
            can_train_vq_only= args.can_train_vq_only, train_vq_only= args.train_vq_only,
            add_asr=args.add_asr, replace_input_by_asr=args.replace_input_by_asr
        )
        return model_config
    elif model_name.upper() == 'FASTPITCHMULTIGENERATED':
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            #other params
            gen_embedding_dim=args.gen_embedding_dim, bottleneck_dim=args.bottleneck_dim,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type,
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim, 
            generated_encoder=bool(args.generated_encoder), generated_decoder=bool(args.generated_decoder), generated_temporal=bool(args.generated_temporal),
            groups=args.batch_size, language_embedding_dim=args.language_embedding_dim, speaker_embedding_dim=args.speaker_embedding_dim,
        )
        return model_config


    elif model_name.upper() == 'FASTPITCHMULTIWITHALIGNER':
        #print(args)
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type, 
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim, 
        )
        return model_config
    elif model_name.upper() == 'FASTSPEECHMODEL':
        #print(args)
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type, 
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim,
            use_fastspeech2_module=args.use_fastspeech2_module,
            use_nemo_module=args.use_nemo_module,
        )
        return model_config
    elif model_name.upper() == 'FASTSPEECHMODELWAVVQ':
        #print(args)
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type,
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim,
            use_fastspeech2_module=args.use_fastspeech2_module,
            use_nemo_module=args.use_nemo_module,
            use_glow_mas=args.use_glow_mas,
            use_mas=args.use_mas,
            use_soft_duration_predictor=args.use_soft,
            use_tgt_dur_in_soft=args.use_tgt_dur_soft,
            use_soft_tacotron_duration_predictor=args.use_soft_tacotron,
            use_diffusion_decoder=args.use_diffusion, use_prior_grad=args.use_priorgrad,
            use_hifigan_wav_decoder=args.use_hifigan_wav_decoder,
            use_msmc_hifigan_wav_decoder=args.use_msmc_hifigan_wav_decoder,
            use_vq_gan=args.use_vq_gan,
            use_vq_frame_decoder=args.use_vq_frame_decoder,
            vq_pred_mel=args.vq_pred_mel,
            vq_use_gan_output_for_mel_generation=args.vq_use_gan_output_for_mel_generation,
            vq_use_fastpitch_transformer=args.vq_use_fastpitch_transformer, use_conformer=args.use_conformer,
            can_train_vq_only=args.can_train_vq_only, train_vq_only=args.train_vq_only
        )
        return model_config
    elif model_name.upper() == 'FASTSPEECHMODELWAVVQTTSASR':
        #print(args)
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type,
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim,
            use_fastspeech2_module=args.use_fastspeech2_module,
            use_nemo_module=args.use_nemo_module,

            use_glow_mas=args.use_glow_mas,
            use_mas=args.use_mas,
            use_soft_duration_predictor=args.use_soft,
            use_tgt_dur_in_soft=args.use_tgt_dur_soft,
            use_soft_tacotron_duration_predictor=args.use_soft_tacotron,
            use_diffusion_decoder=args.use_diffusion, use_prior_grad=args.use_priorgrad,
            use_hifigan_wav_decoder=args.use_hifigan_wav_decoder,
            use_msmc_hifigan_wav_decoder=args.use_msmc_hifigan_wav_decoder,
            use_vq_gan=args.use_vq_gan,
            use_vq_frame_decoder=args.use_vq_frame_decoder,
            vq_pred_mel=args.vq_pred_mel,
            vq_use_gan_output_for_mel_generation=args.vq_use_gan_output_for_mel_generation,
            vq_use_fastpitch_transformer=args.vq_use_fastpitch_transformer, use_conformer=args.use_conformer,
            can_train_vq_only=args.can_train_vq_only, train_vq_only=args.train_vq_only,
            add_asr=args.add_asr, replace_input_by_asr=args.replace_input_by_asr
        )
        return model_config
    elif model_name.upper() == 'FASTPITCHMULTIWITHMIXER':
        #print(args)
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type, 
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim, 
        )
        return model_config

    elif model_name.upper() == 'FASTPITCHMULTIWITHVITSALIGNER':
        #print(args)
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type, 
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim, 
        )
        return model_config

    elif model_name.upper() == 'TACOTRON2':
        #print(args)
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols, #len(get_symbols(args.symbol_set)),
            padding_idx=args.padding_idx, #get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # languages parameters
            n_languages=args.n_languages,
            language_emb_weight=args.language_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
            use_reversal_classifier=bool(args.use_reversal_classifier), reversal_classifier_type=args.reversal_classifier_type, 
            reversal_gradient_clipping= args.reversal_gradient_clipping ,reversal_classifier_dim= args.reversal_classifier_dim, 
        )
        return model_config

    else:
        raise NotImplementedError(model_name)
