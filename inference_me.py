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

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import models
import time
import sys
import warnings
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity


import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write


import torch.cuda.profiler as profiler
import torch.distributed as dist
sys.path.insert(0, '/home/roseline/benit/apex/')
#sys.path.insert(0, '/home/roseline/apex/')

from apex.optimizers import FusedAdam, FusedLAMB
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from common import utils
from common.tb_dllogger import (init_inference_metadata, stdout_metric_format,
                                unique_log_fpath)
#from common.text import cmudict
from common.text.text_processing import TextProcessing
from common.audio_processing import griffin_lim
from common.stft import STFT

from pitch_transform import pitch_transform_custom
from waveglow import model as glow
from waveglow.denoiser import Denoiser
#from common.text import cmudict
from common.utils import prepare_tmp, load_wav, mel_spectrogram, load_checkpoint
from fastpitch.attn_loss_function import AttentionBinarizationLoss
#from fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
#from fastpitch.data_function_me import TTSDatasetMeMulti, TTSCollateMeMulti, batch_to_gpu_me, batch_to_cpu_me

from fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
from fastpitch.data_function_me import TTSDatasetMeMulti, TTSCollateMeMulti, batch_to_gpu_me, batch_to_cpu_me
from fastpitch.data_function_me_char import TTSDatasetMeMultiChar, TTSCollateMeMultiChar, batch_to_gpu_me_char

from fastpitch.loss_function import FastPitchLoss

from hifigan.env import AttrDict
#from hifigan.models import hifigan

import json

#import common.tb_dllogger as logger
#import models
#from common.text import cmudict
#from common.utils import prepare_tmp
#from fastpitch.attn_loss_function import AttentionBinarizationLoss
#from fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
#from fastpitch.data_function_me import TTSDatasetMeMulti, TTSCollateMeMulti, batch_to_gpu_me
#from fastpitch.data_function_me_char import TTSDatasetMeMultiChar, TTSCollateMeMultiChar, batch_to_gpu_me_char
#from fastpitch.loss_function import FastPitchLoss
#from torchsummary import summary

#from coqui_ai.tacotron.losses import TacotronLoss
#from coqui_ai.mdnlosses import AlignTTSLoss
#from  coqui_ai.mixer_tts.mixer_tts import MixerLoss
#from coqui_ai.vits.vits_losses import VitsLoss
#from coqui_ai.fastspeech.fastspeech2_submodules import FastSpeechLoss
#import common.layers as layers

from common.audio_processing import griffin_lim, spectrogram_to_wav_griffin_lim_sav
from common.stft import STFT
import common.tf_griffin_lim as tfgriff

from common.mcd import get_mcd_between_wav_files, get_mcd_between_mel_spectograms, get_mcd_and_penalty_and_final_frame_number


sys.modules['glow'] = glow


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    #parser.add_argument('-i', '--input', type=str, required=True,
    #                    help='Full path to the input text (phareses separated by newlines)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder to save audio (file per phrase)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--save-mels', action='store_true', help='')
    parser.add_argument('--cuda', action='store_true', help='Run inference on a GPU using CUDA')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')
    parser.add_argument('--fastpitch', type=str,
                        help='Full path to the generator checkpoint file (skip to use ground truth mels)')

    parser.add_argument('--waveglow', type=str, default='SKIP',
                        help='Full path to the WaveGlow model checkpoint file (skip to only generate mels)')

    parser.add_argument('--hifigan', type=str, default='SKIP',
                        help='Full path to the hifigan model checkpoint file (skip to only generate mels)')

    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float,
                        help='WaveGlow sigma')
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float,
                        help='WaveGlow denoising')
    parser.add_argument('-sr', '--sampling-rate', default=16000, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=8)
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Warmup iterations before measuring performance')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Repeat inference for benchmarking')
    parser.add_argument('--torchscript', action='store_true',
                        help='Apply TorchScript')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset (for loading extra data fields)')
    parser.add_argument('--speaker', type=int, default=0,
                        help='Speaker ID for a multi-speaker model')
    parser.add_argument('--seed', default=1234, type=int,   help='seed')

    parser.add_argument('--p-arpabet', type=float, default=0.0, help='')
    parser.add_argument('--heteronyms-path', type=str, default='cmudict/heteronyms',
                        help='')
    parser.add_argument('--cmudict-path', type=str, default='cmudict/cmudict-0.7b',
                        help='')
    transform = parser.add_argument_group('transform')
    transform.add_argument('--fade-out', type=int, default=10,
                           help='Number of fadeout frames at the end')
    transform.add_argument('--pace', type=float, default=1.0,
                           help='Adjust the pace of speech')
    transform.add_argument('--pitch-transform-flatten', action='store_true',
                           help='Flatten the pitch')
    transform.add_argument('--pitch-transform-invert', action='store_true',
                           help='Invert the pitch wrt mean value')
    transform.add_argument('--pitch-transform-amplify', type=float, default=1.0,
                           help='Amplify pitch variability, typical values are in the range (1.0, 3.0).')
    transform.add_argument('--pitch-transform-shift', type=float, default=0.0,
                           help='Raise/lower the pitch by <hz>')
    transform.add_argument('--pitch-transform-custom', action='store_true',
                           help='Apply the transform from pitch_transform.py')

    text_processing = parser.add_argument_group('Text processing parameters')
    text_processing.add_argument('--text-cleaners', nargs='*',
                                 default=['english_cleaners'], type=str,
                                 help='Type of text cleaners for input text')
    text_processing.add_argument('--symbol-set', type=str, default='english_basic',
                                 help='Define symbol set for input text')

    myfonfrparam=  parser.add_argument_group('parameter for fongbe french generated model')

    myfonfrparam.add_argument(
        "--use-norm", default=1, type=int, help="usr norm-mels for train or raw."
    )
    myfonfrparam.add_argument(
        "--f0-stat", default="./dump/stats_f0.npy", type=str, help="f0-stat path.",
    )
    myfonfrparam.add_argument(
        "--energy-stat",
        default="./dump/stats_energy.npy",
        type=str,
        help="energy-stat path.",
    )
    #myfonfrparam.add_argument(
    #    "--dataset_config", default="preprocess/libritts_preprocess.yaml", type=str,
    #)
    myfonfrparam.add_argument(
        "--dataset_stats", default="dump/stats.npy", type=str,
    )
    myfonfrparam.add_argument(
        "--dataset_mapping", default="dump/libritts_mapper.npy", type=str,
    )
    myfonfrparam.add_argument(
        "--use_char", default=0, type=int,help="use char instead of phonemes"
    )
    myfonfrparam.add_argument(
        "--tones", default=0, type=int,  help="use separeted tones"
    )
    myfonfrparam.add_argument(
        "--toneseparated", default=0, type=int, help="use separeted tones"
    )
    myfonfrparam.add_argument(
        "--convert_ipa", default=0, type=int,help="used shared ipa phoneme"
    )
    myfonfrparam.add_argument(
        "--use_ipa_phone", default=0, type=int,help="used shared ipa phoneme"
    )
    myfonfrparam.add_argument(
        "--load_language_array", default=0, type=int,help="use parameter generation network"
    )
    myfonfrparam.add_argument(
        "--load_speaker_array", default=0, type=int,help="use parameter generation network"
    )
    
    myfonfrparam.add_argument(
        "--filter_on_lang", default=0, type=int,help="can delete specific languages"
    )
    myfonfrparam.add_argument(
        "--filter_on_speaker", default=0, type=int,help="can delete specific speakers"
    )
    myfonfrparam.add_argument(
        "--filter_on_uttid", default=0, type=int, help="can delete specific utt_ids"
    )
    myfonfrparam.add_argument(
        "--filter_remove_lang_ids", default=None, type=str, help="comma separated list of ids of language to delete"
    )
    myfonfrparam.add_argument(
        "--filter_remove_speaker_ids", default=None, type=str,help="comma separated list of ids of speaker to delete"
    )
    myfonfrparam.add_argument(
        "--filter_remove_uttid_ids", default=None, type=str,help="comma separated list of ids of utt to delete"
    )
    myfonfrparam.add_argument(
        "--mtype", type=int,default=0, required=False, help="type de fastspeech."
    )
    myfonfrparam.add_argument(
        "--generated", default=0, type=int,help="use parameter generation network"
    )
    #myfonfrparam.add_argument(
    #    "--balanced-sampling", default=1, type=int,help="balanced sampling"
    #)
    #myfonfrparam.add_argument(
    #     "--perfect-sampling", default=1, type=int,help="perfect sampling"
    # )
    #myfonfrparam.add_argument(
    #    "--balanced-training", default=0, type=int,help="balanced training"
    #)


    myfonfrparam.add_argument(
        "--variant", default="", type=str,
    )

    myfonfrparam.add_argument('--train-type', default='fastpitch',
                      choices=['fastpitch','tensorflowtts', "create"],
                      help='using fastspeech data or tensorflowtts data')

    myfonfrparam.add_argument('--format', default='npy',
                      choices=['npy','pt'],
                      help='using fastspeech data npy or tensorflowtts data npy')


    cond = parser.add_argument_group('conditioning on additional attributes')
    #cond.add_argument('--n-speakers', type=int, default=1,
    #                  help='Number of speakers in the model.')

    #cond.add_argument('--classifier-loss-scale', type=float,
    #                 default=0.5, help='Rescale alignment loss')
    #cond.add_argument('--reversal-classifier', action='store_true',
    #                  help='use reversal classifier')
    #cond.add_argument('--reversal-classifier-type', type=str,
    #                 default="reversal", help='reversal classifier type')
    #cond.add_argument('--reversal-classifier-w', type=float,
    #                 default=1.0, help='Rescale alignment loss')



    #cond.add_argument(  "--train-dir",  default="dump/train", type=str,help="directory including training data. ",    )
    #cond.add_argument(   "--dev-dir", default="dump/valid",  type=str,  help="directory including development data. ",    )


    cond.add_argument('--pitch-online-method', default='pyin',
                      choices=['pyin'],
                      help='Calculate pitch on the fly during trainig')
    cond.add_argument('--pitch-online-dir', type=str, default=None,
                      help='A directory for storing pitch calculated on-line')
    cond.add_argument('--pitch-mean', type=float, default=214.72203,
                      help='Normalization value for pitch')
    cond.add_argument('--pitch-std', type=float, default=65.72038,
                      help='Normalization value for pitch')

    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--max-wav-value', default=32768.0, type=float,
                       help='Maximum audiowave value')
    #audio.add_argument('--sampling-rate', default=16000, type=int,
    #                   help='Sampling rate')
    audio.add_argument('--filter-length', default=1024, type=int,
                       help='Filter length')
    audio.add_argument('--hop-length', default=256, type=int,
                       help='Hop (stride) length')
    audio.add_argument('--win-length', default=1024, type=int,
                       help='Window length')
    audio.add_argument('--mel-fmin', default=0.0, type=float,
                       help='Minimum mel frequency')
    audio.add_argument('--mel-fmax', default=8000.0, type=float,
                       help='Maximum mel frequency')
    audio.add_argument('--nmel-channels', default=80, type=int,
                       help='Maximum mel frequency')


    #symbols = parser.add_argument_group('symbols parameters')
    #symbols.add_argument('--n-symbolss', default=300, type=int,
    #                     help='Number of symbols in dictionary')
    #symbols.add_argument('--padding-idxs', default=0, type=int,
    #                     help='Index of padding symbol in dictionary')
    #symbols.add_argument('--symbols-embedding-dims', default=384, type=int,
    #                     help='Input embedding dimension')
    #symbols.add_argument('--language-embedding-dims', default=8, type=int,
    #                     help='Input embedding dimension')
    #symbols.add_argument('--speaker-embedding-dims', default=48, type=int,
    #                     help='Input embedding dimension')


    return parser


def load_model_from_ckpt(checkpoint_path, ema, model):

    checkpoint_data = torch.load(checkpoint_path)
    status = ''

    if 'state_dict' in checkpoint_data:
        sd = checkpoint_data['state_dict']
        if ema and 'ema_state_dict' in checkpoint_data:
            sd = checkpoint_data['ema_state_dict']
            status += ' (EMA)'
        elif ema and not 'ema_state_dict' in checkpoint_data:
            print(f'WARNING: EMA weights missing for {checkpoint_data}')

        if any(key.startswith('module.') for key in sd):
            sd = {k.replace('module.', ''): v for k,v in sd.items()}
        status += ' ' + str(model.load_state_dict(sd, strict=False))
    else:
        model = checkpoint_data['model']
    print(f'Loaded {checkpoint_path}{status}')

    return model


def load_hifigan_model(checkpoint_path, ema, generator, device):
    state_dict_g = load_checkpoint(checkpoint_path, device)
    print("###state_dict_g ", state_dict_g.keys())
    generator.load_state_dict(state_dict_g['generator'], strict=False)
    return generator

def load_hifigan_model_from_path(checkpoint_path, ema, generator, mpd, msd,  device):
    if os.path.isdir(checkpoint_path):
        cp_g = scan_checkpoint(h.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(h.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'], strict=False)
        mpd.load_state_dict(state_dict_do['mpd'], strict=False)
        msd.load_state_dict(state_dict_do['msd'], strict=False)
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
    return generator


def load_waveglow_checkpoint(checkpoint_path, model,  device): #optimizer,
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    iteration = checkpoint_dict['iteration']
    #optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model #, optimizer, iteration

def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[],  args=None, forward_is_infer=False, ema=True,
                         jitable=False):

    #print(model_name,"  args=", args, type(args))
    #print("\n\nunk_args = ",unk_args, type(unk_args))
    if args.variant is not None and args.variant!="":
        if args.variant.lower()=="fastspeechmodel":
            model_parser = models.parse_model_args('FastSpeechModel', parser, add_help=False)
        elif args.variant.lower()=="fastpitchmultiwithaligner":
            model_parser = models.parse_model_args('FastPitchMultiWithAligner', parser, add_help=False)
        elif args.variant.lower()=="fastpitchmultiwithmixer":
            model_parser = models.parse_model_args('FastPitchMultiWithMixer', parser, add_help=False)
        elif args.variant.lower()=="fastpitchmultiwithvitsaligner":
            model_parser = models.parse_model_args('FastPitchMultiWithVITSAligner', parser, add_help=False)
        elif args.variant.lower()=="tacotron2":
            model_parser = models.parse_model_args('Tacotron2', parser, add_help=False)
        else:
            raise ValueError(f'not yet implemented')
    else:
        model_parser = models.parse_model_args(model_name, parser, add_help=False)

    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))
    #print("\n\n ### "+model_name+" model_args = ",model_args, type(model_args))
    #print("\n\n model_unk_args = ",model_unk_args, type(model_unk_args))
    model_args.n_speakers=args.n_speakers
    model_args.n_languages=args.n_languages
    #model_args.n_symbols=args.n_symbolss
    #model_args.padding_idx=args.padding_idxs
    #model_args.symbols_embedding_dim=args.symbols_embedding_dims
    #model_args.language_embedding_dim=args.language_embedding_dims
    #model_args.speaker_embedding_dim=args.speaker_embedding_dims



    if args.variant is not None and args.variant!="":
        if args.variant.lower()=="fastspeechmodel":
            model_config = models.get_model_config('FastSpeechModel', model_args)
        elif args.variant.lower()=="fastpitchmultiwithaligner":
            model_config = models.get_model_config('FastPitchMultiWithAligner', model_args)
        elif args.variant.lower()=="fastpitchmultiwithmixer":
            model_config = models.get_model_config('FastPitchMultiWithMixer', model_args)
        elif args.variant.lower()=="fastpitchmultiwithvitsaligner":
            model_config = models.get_model_config('FastPitchMultiWithVITSAligner', model_args)
        elif args.variant.lower()=="tacotron2":
            model_config = models.get_model_config('Tacotron2', model_args)
        else:
            raise ValueError(f'not yet implemented')
    elif   "FastPitch" in model_name:
        if args.generated:
            model_config = models.get_model_config('FastPitchMultiGenerated', model_args)
        else:
            model_config = models.get_model_config('FastPitchMulti', model_args)
        if args.tones:
            raise ValueError(f'not yet implemented')
        if args.toneseparated:
            raise ValueError(f'not yet implemented')
    else:
        model_config = models.get_model_config(model_name, model_args)

        if args.variant is not None and args.variant!="":
        if args.variant.lower()=="fastspeechmodel":
            model = models.get_model('FastSpeechModel',  model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)
        elif args.variant.lower()=="fastpitchmultiwithaligner":
            model = models.get_model('FastPitchMultiWithAligner', model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitablee)
        elif args.variant.lower()=="fastpitchmultiwithmixer":
            model = models.get_model('FastPitchMultiWithMixer', model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)
        elif args.variant.lower()=="fastpitchmultiwithvitsaligner":
            model = models.get_model('FastPitchMultiWithVITSAligner', model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)
        elif args.variant.lower()=="tacotron2":
            model = models.get_model('Tacotron2', model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)
        else:
            raise ValueError(f'not yet implemented')
    elif args.generated:
        model = models.get_model('FastPitchMultiGenerated', model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)
    elif args.tones:
        raise ValueError(f'not yet implemented')
    elif args.toneseparated:
        raise ValueError(f'not yet implemented')
    else:
        model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)

    if checkpoint is not None:
        if model_name.upper() == 'HIFIGAN' or model_name.upper() == 'HIFI-GAN' or model_name.upper() == 'HI-FI-GAN':
            model=load_hifigan_model(checkpoint, ema, model, device)
        else:
            model = load_model_from_ckpt(checkpoint, ema, model)

    if model_name == "WaveGlow":
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability

        model = model.remove_weightnorm(model)

    #if model_name.upper() == 'HIFIGAN' or model_name.upper() == 'HIFI-GAN' or model_name.upper() == 'HI-FI-GAN':
    #    model = model.remove_weight_norm()

    if amp:
        model.half()
    model.eval()
    return model.to(device), model_parser, model_args


def load_fields(fpath):
    lines = [l.strip() for l in open(fpath, encoding='utf-8')]
    if fpath.endswith('.tsv'):
        columns = lines[0].split('\t')
        fields = list(zip(*[t.split('\t') for t in lines[1:]]))
    else:
        columns = ['text']
        fields = [lines]
    return {c:f for c, f in zip(columns, fields)}


def prepare_input_sequence(fields, device, symbol_set, text_cleaners,
                           batch_size=128, dataset=None, load_mels=False,
                           load_pitch=False, p_arpabet=0.0):
    tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet)

    fields['text'] = [torch.LongTensor(tp.encode_text(text))
                      for text in fields['text']]
    order = np.argsort([-t.size(0) for t in fields['text']])

    fields['text'] = [fields['text'][i] for i in order]
    fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])

    for t in fields['text']:
        print(tp.sequence_to_text(t.numpy()))

    if load_mels:
        assert 'mel' in fields
        fields['mel'] = [
            torch.load(Path(dataset, fields['mel'][i])).t() for i in order]
        fields['mel_lens'] = torch.LongTensor([t.size(0) for t in fields['mel']])

    if load_pitch:
        assert 'pitch' in fields
        fields['pitch'] = [
            torch.load(Path(dataset, fields['pitch'][i])) for i in order]
        fields['pitch_lens'] = torch.LongTensor([t.size(0) for t in fields['pitch']])

    if 'output' in fields:
        fields['output'] = [fields['output'][i] for i in order]

    # cut into batches & pad
    batches = []
    for b in range(0, len(order), batch_size):
        batch = {f: values[b:b+batch_size] for f, values in fields.items()}
        for f in batch:
            if f == 'text':
                batch[f] = pad_sequence(batch[f], batch_first=True)
            elif f == 'mel' and load_mels:
                batch[f] = pad_sequence(batch[f], batch_first=True).permute(0, 2, 1)
            elif f == 'pitch' and load_pitch:
                batch[f] = pad_sequence(batch[f], batch_first=True)

            if type(batch[f]) is torch.Tensor:
                batch[f] = batch[f].to(device)
        batches.append(batch)

    return batches


def build_pitch_transformation(args):
    if args.pitch_transform_custom:
        def custom_(pitch, pitch_lens, mean, std):
            return (pitch_transform_custom(pitch * std + mean, pitch_lens)
                    - mean) / std
        return custom_

    fun = 'pitch'
    if args.pitch_transform_flatten:
        fun = f'({fun}) * 0.0'
    if args.pitch_transform_invert:
        fun = f'({fun}) * -1.0'
    if args.pitch_transform_amplify:
        ampl = args.pitch_transform_amplify
        fun = f'({fun}) * {ampl}'
    if args.pitch_transform_shift != 0.0:
        hz = args.pitch_transform_shift
        fun = f'({fun}) + {hz} / std'
    return eval(f'lambda pitch, pitch_lens, mean, std: {fun}')


class MeasureTime(list):
    def __init__(self, *args, cuda=True, **kwargs):
        super(MeasureTime, self).__init__(*args, **kwargs)
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.append(time.perf_counter() - self.t0)

    def __add__(self, other):
        assert len(self) == len(other)
        return MeasureTime((sum(ab) for ab in zip(self, other)), cuda=cuda)


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()

    #if args.p_arpabet > 0.0:
    #    cmudict.initialize(args.cmudict_path, keep_ambiguous=True)

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.output is not None:
        Path(args.output).mkdir(parents=False, exist_ok=True)

    log_fpath = args.log_file or str(Path(args.output, 'nvlog_infer.json'))
    log_fpath = unique_log_fpath(log_fpath)
    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_fpath),
                            StdOutBackend(Verbosity.VERBOSE,
                                          metric_format=stdout_metric_format)])
    init_inference_metadata()
    [DLLogger.log("PARAMETER", {k: v}) for k, v in vars(args).items()]

    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')



    #args.mixed_precision = bool(args.mixed_precision)
    args.use_norm = bool(args.use_norm)
    args.tones= bool(args.tones)
    args.toneseparated= bool(args.toneseparated)
    args.generated= bool(args.generated)
    args.use_char=bool(args.use_char)
    args.convert_ipa=bool(args.convert_ipa)
    args.use_ipa_phone= bool(args.use_ipa_phone)
    args.load_speaker_array= bool(args.load_speaker_array)
    args.load_language_array= bool(args.load_language_array)

    args.filter_on_lang=bool(args.filter_on_lang)
    args.filter_on_speaker=bool(args.filter_on_speaker)
    args.filter_on_uttid=bool(args.filter_on_uttid)
    #print("### args args.filter_remove_lang_ids ", args.filter_remove_lang_ids)
    l_filter_remove_lang_ids=args.filter_remove_lang_ids
    l_filter_remove_speaker_ids=args.filter_remove_speaker_ids
    l_filter_remove_uttid_ids=args.filter_remove_uttid_ids

    if args.filter_remove_lang_ids is not None:
        l_filter_remove_lang_ids=args.filter_remove_lang_ids=[int(i) if str(i).isnumeric() else i   for i in args.filter_remove_lang_ids.split(',')] 
    if args.filter_remove_speaker_ids is not None:
        l_filter_remove_speaker_ids=args.filter_remove_speaker_ids=[int(i) if str(i).isnumeric() else i for i in args.filter_remove_speaker_ids.split(',')] 
    if args.filter_remove_uttid_ids is not None:
        l_filter_remove_uttid_ids=args.filter_remove_uttid_ids=[i for i in args.filter_remove_uttid_ids.split(',')]


    use_tones=False
    if args.tones or args.toneseparated:
        use_tones=True


    format=args.format
    #format='npy'

    if format == "npy":
        wave_query= "*-wave.npy"
        charactor_query = "*-ids.npy"
        tones_query = "*-tonesids.npy"

        speaker_query = "*-speaker-ids.npy"
        language_query = "*-language-ids.npy"
        language_speaker_query = "*-language-speaker-ids.npy"

        language_array_query="*-languagearray-ids.npy"
        speaker_array_query="*-speakerarray-ids.npy"

        mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"

        duration_query = "*-durations.npy"
        f0_query = "*-raw-f0.npy"
        pitch_query = "*.pt"
        energy_query = "*-raw-energy.npy"
        speaker_load_fn = np.load
        language_load_fn = np.load
    elif format=="pt":
        #wave_query= "*-wave.pt"
        #charactor_query = "*-ids.pt"
        #tones_query = "*-tonesids.pt"
        wave_query= "*-wave.npy"
        charactor_query = "*-ids.npy"
        tones_query = "*-tonesids.npy"

        #speaker_query = "*-speaker-ids.pt"
        #language_query = "*-language-ids.pt"
        #language_speaker_query = "*-language-speaker-ids.pt"
        speaker_query = "*-speaker-ids.npy"
        language_query = "*-language-ids.npy"
        language_speaker_query = "*-language-speaker-ids.npy"

        #language_array_query="*-languagearray-ids.pt"
        #speaker_array_query="*-speakerarray-ids.pt"
        language_array_query="*-languagearray-ids.npy"
        speaker_array_query="*-speakerarray-ids.npy"

        #mel_query = "*-raw-feats.pt" if args.use_norm is False else "*-norm-feats.pt"
        mel_query = "*.pt"

        duration_query = "*-durations.pt"
        f0_query = "*-raw-f0.pt"
        energy_query = "*-raw-energy.pt"
        speaker_load_fn = torch.load
        language_load_fn = torch.load
    else:
        raise ValueError("Only npy are supported.")



    load_mel_from_disk=False
    load_my_mel_from_disk=False
    load_my_duration_from_disk= False
    load_pitch_from_disk=False
    load_energy_from_disk=False
    load_f0_from_disk=False
    load_alignment_prior_from_disk=False

    if args.train_type.lower()== 'fastpitch':
        wave_query= "*-wave.npy"
        charactor_query = "*-ids.npy"
        tones_query = "*-tonesids.npy"

        speaker_query = "*-speaker-ids.npy"
        language_query = "*-language-ids.npy"
        language_speaker_query = "*-language-speaker-ids.npy"

        language_array_query="*-languagearray-ids.npy"
        speaker_array_query="*-speakerarray-ids.npy"

        mel_query = "*.pt"

        duration_query = "*.pt"
        f0_query = "*-raw-f0.npy"
        pitch_query = "*.pt"
        energy_query = "*-raw-energy.npy"
        speaker_load_fn = np.load
        language_load_fn = np.load
        load_f0_from_disk=False
        load_pitch_from_disk=True
        load_mel_from_disk=True
        load_alignment_prior_from_disk=True
        mel_load_fn = torch.load
        duration_load_fn=torch.load
        f0_load_fn=torch.load
        energy_load_fn=torch.load
    elif args.train_type.lower()== 'create':
        wave_query= "*-wave.npy"
        charactor_query = "*-ids.npy"
        tones_query = "*-tonesids.npy"

        speaker_query = "*-speaker-ids.npy"
        language_query = "*-language-ids.npy"
        language_speaker_query = "*-language-speaker-ids.npy"

        language_array_query="*-languagearray-ids.npy"
        speaker_array_query="*-speakerarray-ids.npy"

        mel_query = "*.pt"

        duration_query = "*.pt"
        f0_query = "*-raw-f0.npy"
        pitch_query = "*.pt"
        energy_query = "*-raw-energy.npy"
        speaker_load_fn = np.load
        language_load_fn = np.load
        load_f0_from_disk=False
        load_pitch_from_disk=False
        load_mel_from_disk=False
        mel_load_fn = torch.load
        duration_load_fn=torch.load
        f0_load_fn=torch.load
        energy_load_fn=torch.load
    elif args.train_type.lower()== 'tensorflowtts':
        wave_query= "*-wave.npy"
        charactor_query = "*-ids.npy"
        tones_query = "*-tonesids.npy"

        speaker_query = "*-speaker-ids.npy"
        language_query = "*-language-ids.npy"
        language_speaker_query = "*-language-speaker-ids.npy"

        language_array_query="*-languagearray-ids.npy"
        speaker_array_query="*-speakerarray-ids.npy"

        mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"

        duration_query =  "*-durations.npy"
        f0_query = "*-raw-f0.npy"
        pitch_query = "*.pt"
        energy_query = "*-raw-energy.npy"
        speaker_load_fn = np.load
        language_load_fn = np.load
        load_f0_from_disk=True
        load_pitch_from_disk=False
        load_mel_from_disk=True
        load_my_mel_from_disk=True
        load_my_duration_from_disk= True
        load_energy_from_disk=True
        mel_load_fn = np.load
        duration_load_fn=np.load
        f0_load_fn=np.load
        energy_load_fn=np.load
        

    speakers_map = None
    languages_map = None

    with open(args.dataset_mapping) as f:
        dataset_mapping = json.load(f)
        speakers_map = dataset_mapping["speakers_map"]
        languages_map = dataset_mapping["languages_map"]

    args.n_speakers=n_speakers=len(speakers_map)+10
    args.n_languages=n_languages= len(languages_map)+3
    args.n_mel_channels=args.nmel_channels

    model_args= args

    if args.fastpitch != 'SKIP':
        if args.generated:
            generator,model_parser, model_args = load_and_setup_model(
                'FastPitchMultiGenerated', parser, args.fastpitch, args.amp, device,
                unk_args=unk_args, args=args, forward_is_infer=True, ema=args.ema,
                jitable=args.torchscript)

            if args.torchscript:
                generator = torch.jit.script(generator)

        else:
            generator, model_parser, model_args = load_and_setup_model(
                'FastPitchMulti', parser, args.fastpitch, args.amp, device,
                unk_args=unk_args, args=args, forward_is_infer=True, ema=args.ema,
                jitable=args.torchscript)
    
            if args.torchscript:
                generator = torch.jit.script(generator)
        #else:
        #    generator, model_parser, model_args  = load_and_setup_model(
        #        'FastPitch', parser, args.fastpitch, args.amp, device,
        #        unk_args=unk_args, args=args, forward_is_infer=True, ema=args.ema,
        #        jitable=args.torchscript)

        #   if args.torchscript:
        #        generator = torch.jit.script(generator)
 
    else:
        generator = None

    if args.tones:
        raise ValueError(f'not yet implemented')
    if args.toneseparated:
        raise ValueError(f'not yet implemented')


    #print("model summary ")
    #print(generator)


    generator.pitch_mean[0] = args.pitch_mean
    generator.pitch_std[0] = args.pitch_std

    convert_ipa=False
    if args.convert_ipa:
        convert_ipa=True
    if args.use_ipa_phone:
        convert_ipa=False



    #fields = load_fields(args.input)
    #batches = prepare_input_sequence(
    #    fields, device, args.symbol_set, args.text_cleaners, args.batch_size,
    #    args.dataset_path, load_mels=(generator is None), p_arpabet=args.p_arpabet)

    valset = TTSDatasetMeMulti(            root_dir=args.dataset_path,
            wave_query=wave_query, charactor_query=charactor_query, tones_query=tones_query,
            speaker_query=speaker_query,
            language_speaker_query=language_speaker_query,
            language_query=language_query,
            charactor_load_fn=np.load, language_load_fn=language_load_fn,
            speaker_load_fn=speaker_load_fn,
            mel_query=mel_query, mel_load_fn=mel_load_fn,
            duration_query=duration_query, duration_load_fn=duration_load_fn,
            f0_query=f0_query, f0_load_fn=f0_load_fn, pitch_query=pitch_query,
            energy_query=energy_query, energy_load_fn=energy_load_fn,
            f0_stat=args.f0_stat,
            energy_stat=args.energy_stat,
            #mel_length_threshold=mel_length_threshold, max_mel_length_threshold=max_mel_length_threshold,
            speakers_map=speakers_map,
            languages_map=languages_map, convert_ipa=args.convert_ipa, use_tones=use_tones,
            filter_on_lang=args.filter_on_lang,
            filter_on_speaker=args.filter_on_speaker,
            filter_on_uttid=args.filter_on_uttid,
            filter_remove_lang_ids=l_filter_remove_lang_ids,        
            filter_remove_speaker_ids=l_filter_remove_speaker_ids,   
            filter_remove_uttid_ids=l_filter_remove_uttid_ids, 
            load_language_array=args.load_language_array , load_speaker_array=args.load_speaker_array,
            language_array_query=language_array_query,   speaker_array_query=speaker_array_query,

                # config=None,
                n_speakers=n_speakers, n_languages=n_languages,
                
                #text_cleaners=['english_cleaners_v2'],
                n_mel_channels=args.nmel_channels,
                p_arpabet=args.p_arpabet,
                ####n_speakers=args.n_speakers,
                load_mel_from_disk=load_mel_from_disk, load_my_mel_from_disk=load_my_mel_from_disk, load_my_duration_from_disk= load_my_duration_from_disk,
                load_pitch_from_disk=load_pitch_from_disk, load_energy_from_disk=load_energy_from_disk, load_f0_from_disk=load_f0_from_disk,
                load_alignment_prior_from_disk=load_alignment_prior_from_disk,
                pitch_mean=args.pitch_mean,
                pitch_std=args.pitch_std,
                max_wav_value=args.max_wav_value,
                sampling_rate=args.sampling_rate,
                filter_length=args.filter_length,
                hop_length=args.hop_length,
                win_length=args.win_length,
                mel_fmin=args.mel_fmin,
                mel_fmax=args.mel_fmax,
                betabinomial_online_dir=None,
                pitch_online_dir=None,
                pitch_online_method=args.pitch_online_method , #f0_method, 
             )#**vars(args)


    #if  distributed_run:
    #    valid_sampler, shuffle = DistributedSampler(trainset), False
    #else:
    valid_sampler, shuffle = None, False


    collate_fn = TTSCollateMeMulti()

    mstft= STFT(filter_length=args.filter_length, hop_length=args.hop_length, win_length=args.win_length, window='hann')
    mstft2 = layers.TacotronSTFT(
                filter_length=args.filter_length, hop_length=args.hop_length, win_length=args.win_length,
                n_mel_channels=args.nmel_channels, sampling_rate=args.sampling_rate, mel_fmin=args.mel_fmin, mel_fmax=args.mel_fmax)
    # 4 workers are optimal on DGX-1 (from epoch 2 onwards)
    val_loader = DataLoader(valset, num_workers=1, shuffle=shuffle,
                              sampler=valid_sampler, batch_size=args.batch_size,
                              pin_memory=True, persistent_workers=True,
                              drop_last=True, collate_fn=collate_fn)



    if args.waveglow != 'SKIP':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveglow, wmodel_parser, wmodel_args  = load_and_setup_model(
                'WaveGlow', parser, args.waveglow, args.amp, device,
                unk_args=unk_args,  args=model_args, forward_is_infer=True, ema=args.ema)
        denoiser = Denoiser(waveglow).to(device)
        waveglow = getattr(waveglow, 'infer', waveglow)
    else:
        waveglow = None

    if args.hifigan != 'SKIP':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hifigan, hifimodel_parser, hifimodel_args  = load_and_setup_model(
                'HIFIGAN', parser, args.hifigan, False #args.amp
                 , device,
                unk_args=unk_args,  args=model_args, forward_is_infer=True, ema=args.ema)
        #denoiser = Denoiser(hifigan).to(device)
        #hifigan = getattr(hifigan, 'infer', hifigan)
    else:
        hifigan = None

    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')



    # Use real data rather than synthetic - FastPitch predicts len
    b= iter(val_loader)

    for _ in tqdm(range(args.warmup_steps), 'Warmup'):
        x, y, num_frames = batch_to_gpu_me(b, use_tones=use_tones)
        with torch.no_grad():
            #if distributed_run:
            #    val_loader.sampler.set_epoch(0)
            if generator is not None:
                #b = batches[0]
                if  use_tones:
                    mel, *_ = generator(inputs=x[0], speaker=x[7], language=x[8]) 
                                    #infer(self, inputs, pace=1.0, dur_tgt=None, pitch_tgt=None, energy_tgt=None, pitch_transform=None, max_duration=75,
                else:
                    mel, *_ = generator(inputs=x[0], speaker=x[6], language=x[7])
            if waveglow is not None:
                audios = waveglow(mel, sigma=args.sigma_infer).float()
                _ = denoiser(audios, strength=args.denoising_strength)
            if hifigan is not None:
                hf_audios = hifigan(mel).float()


    gen_measures = MeasureTime(cuda=args.cuda and torch.cuda.is_available())
    waveglow_measures = MeasureTime(cuda=args.cuda and torch.cuda.is_available())
    hifigan_measures = MeasureTime(cuda=args.cuda and torch.cuda.is_available())

    gen_kw = {'pace': args.pace,
              #'speaker': args.speaker,
              'pitch_tgt': None,
              'pitch_transform': build_pitch_transformation(args)}

    if args.torchscript:
        gen_kw.pop('pitch_transform')
        print('NOTE: Pitch transforms are disabled with TorchScript')

    all_utterances = 0
    all_samples = 0
    all_letters = 0
    all_frames = 0

    hf_all_utterances = 0
    hf_all_samples = 0
    hf_all_letters = 0
    hf_all_frames = 0

    reps = args.repeats
    log_enabled = reps == 1
    log = lambda s, d: DLLogger.log(step=s, data=d) if log_enabled else None

    #del val_loader
    #val_loader = DataLoader(valset, num_workers=1, shuffle=shuffle,
    #                          sampler=valid_sampler, batch_size=args.batch_size,
    #                          pin_memory=True, persistent_workers=True,
    #                          drop_last=True, collate_fn=collate_fn)

    for rep in (tqdm(range(reps), 'Inference') if reps > 1 else range(reps)):
        for i,b in enumerate(val_loader):
            x, y, num_frames = batch_to_gpu_me(b, use_tones=use_tones)
            speaker= x[7] if  use_tones else  x[6]
            language= x[8]  if  use_tones else x[7]
            utt_id= x[10]  if  use_tones else x[9]
            if args.speaker is not None and args.speaker!="" and args.speaker>= 0:
                speaker= args.speaker
            if generator is None:
                log(rep, {'Synthesizing from ground truth mels'})
                mel, mel_lens = x[3], x[4]  if  use_tones else  x[2], x[3] #b['mel'], b['mel_lens']
            else:
                with torch.no_grad(), gen_measures:
                    mel, mel_lens, *_ = generator(inputs=x[0], speaker=speaker, language=language, **gen_kw)  
                    
                gen_infer_perf = mel.size(0) * mel.size(2) / gen_measures[-1]
                all_letters += x[2].sum().item()  if  use_tones else   x[1].sum().item()   #b['text_lens'].sum().item()
                all_frames += mel.size(0) * mel.size(2)
                log(rep, {"fastpitch_frames/s": gen_infer_perf})
                log(rep, {"fastpitch_latency": gen_measures[-1]})

                if args.save_mels:
                    for i, mel_ in enumerate(mel):
                        m = mel_[:, :mel_lens[i].item()].permute(1, 0)
                        fname = f'{utt_id[i]}_mel_{i}.npy' #b['output'][i] if 'output' in b else f'mel_{i}.npy'
                        mel_path = Path(args.output, Path(fname).stem + '.npy')
                        np.save(mel_path, m.cpu().numpy())

            if waveglow is not None and False:
                with torch.no_grad(), waveglow_measures:
                    print("######mel shape ",mel.size())
                    audios = waveglow(mel, sigma=args.sigma_infer)
                    audios = denoiser(audios.float(),
                                      strength=args.denoising_strength
                                      ).squeeze(1)
                    print("######mel shape ",mel.size(), " audio size= ", audios.size())

                all_utterances += len(audios)
                all_samples += sum(audio.size(0) for audio in audios)
                waveglow_infer_perf = (
                    audios.size(0) * audios.size(1) / waveglow_measures[-1])

                log(rep, {"waveglow_samples/s": waveglow_infer_perf})
                log(rep, {"waveglow_latency": waveglow_measures[-1]})

                if args.output is not None and reps == 1:
                    for i, audio in enumerate(audios):
                        audio = audio[:mel_lens[i].item() * args.hop_length]

                        if args.fade_out:
                            fade_len = args.fade_out * args.hop_length
                            fade_w = torch.linspace(1.0, 0.0, fade_len)
                            audio[-fade_len:] *= fade_w.to(audio.device)

                        audio = audio / torch.max(torch.abs(audio))
                        fname = f'{utt_id[i]}_1_audio_waveglow_{i}.wav' #b['output'][i] if 'output' in b else f'audio_{i}.wav'
                        audio_path = Path(args.output, fname)
                        write(audio_path, args.sampling_rate, audio.cpu().numpy())

            if waveglow is not None:
                with torch.no_grad(), waveglow_measures:
                    print("######mel shape ",mel.size())


                    #mel = np.load(Path(args.output, Path(f'{utt_id[i]}_mel_{i}.npy').stem + '.npy'))
                    #mel= torch.from_numpy(np.asarray(mel)).float().permute(1, 0)
                    mel = torch.autograd.Variable(mel.cuda())
                    #mel = torch.unsqueeze(mel, 0)
                    #mel = mel.half() if is_fp16 else mel

                    audios = waveglow(mel, sigma=args.sigma_infer)
                    audios = denoiser(audios.float(), strength=args.denoising_strength  ).squeeze(1)
                    print("######mel shape ",mel.size(), " audio size= ", audios.size())

                all_utterances += len(audios)
                all_samples += sum(audio.size(0) for audio in audios)
                waveglow_infer_perf = ( audios.size(0) * audios.size(1) / waveglow_measures[-1])

                log(rep, {"waveglow_samples/s": waveglow_infer_perf})
                log(rep, {"waveglow_latency": waveglow_measures[-1]})

                if args.output is not None and reps == 1:
                    for i, audio in enumerate(audios):
                        audio = audio.squeeze()[:mel_lens[i].item() * args.hop_length]

                        if args.fade_out:
                            print("####fade ")
                            fade_len = args.fade_out * args.hop_length
                            fade_w = torch.linspace(1.0, 0.0, fade_len)
                            audio[-fade_len:] *= fade_w.to(audio.device)

                        if np.abs(audio).max() >= 1.0:
                            audio = audio / torch.max(torch.abs(audio)) #audio_norm = audio / self.max_wav_value
                        
                        #audio = audio * args.max_wav_value
                        audio = audio.squeeze()
                        audio = audio.cpu().numpy()
                        audio = audio.astype('int16')

                        fname = f'{utt_id[i]}_2_audio_waveglow_{i}.wav' #b['output'][i] if 'output' in b else f'audio_{i}.wav'
                        audio_path = Path(args.output, fname)
                        write(audio_path, args.sampling_rate, audio)#.cpu().numpy()

            if hifigan is not None:
                with torch.no_grad(), hifigan_measures:
                    print("######mel shape ",mel.size())
                    for i, mel_ in enumerate(mel):
                        hf_audios = hifigan(torch.unsqueeze(mel_, 0)).float() #, sigma=args.sigma_infer)
                        #hf_audios = hifigan(torch.unsqueeze(mel_, 0)).float() #, sigma=args.sigma_infer)
                        #hf_audios = denoiser(hf_audios.float(), strength=args.denoising_strength ).squeeze(1)
                        #print("######mel shape ",mel_.size(), torch.unsqueeze(mel_, 0).size(), " audio size= ", hf_audios.size())

                        hf_audios = torch.squeeze(hf_audios,1)
                        hf_all_utterances += len(hf_audios)
                        hf_all_samples += sum(audio.size(0) for audio in hf_audios)
                        #print("###hf_audios ", hf_audios.size())
                        #print("###hifigan_measures ", len(hifigan_measures))
                        last_mes= 1 if (len(hifigan_measures)==0) else hifigan_measures[-1]
                        hifigan_infer_perf = ( hf_audios.size(0) * hf_audios.size(1) / last_mes ) 

                        log(rep, {"hifigan_samples/s": hifigan_infer_perf})
                        log(rep, {"hifigan_latency": last_mes}) #hifigan_measures[-1]

                        if args.output is not None and reps == 1:
                            #for i, audio in enumerate(hf_audios):
                            audio = hf_audios
                            audio = audio.squeeze()
                            audio = audio[:mel_lens[i].item() * args.hop_length]
                            #audio = audio * args.max_wav_value
                            audio = audio.cpu().numpy().astype('int16')


                            #if args.fade_out:
                            #    fade_len = args.fade_out * args.hop_length
                            #    fade_w = torch.linspace(1.0, 0.0, fade_len)
                            #    audio[-fade_len:] *= fade_w.to(audio.device)
                            #audio = audio / torch.max(torch.abs(audio))
                            fname = f'{utt_id[i]}_audio_hifigan_{i}.wav' #b['output'][i] if 'output' in b else f'audio_{i}.wav'
                            audio_path = Path(args.output, fname)
                            write(audio_path, args.sampling_rate, audio)


            if hifigan is not None and False:
                with torch.no_grad(), hifigan_measures:
                    print("######mel shape ",mel.size())
                    hf_audios = hifigan(mel) #, sigma=args.sigma_infer)
                    #hf_audios = denoiser(hf_audios.float(), strength=args.denoising_strength ).squeeze(1)
                    print("######mel shape ",mel.size(), " audio size= ", hf_audios.size())

                hf_all_utterances += len(hf_audios)
                hf_all_samples += sum(audio.size(0) for audio in hf_audios)
                print("###hf_audios ", hf_audios.size())
                print("###hifigan_measures ", len(hifigan_measures))
                hifigan_infer_perf = (
                    hf_audios.size(0) * hf_audios.size(1) / hifigan_measures[-1])

                log(rep, {"hifigan_samples/s": hifigan_infer_perf})
                log(rep, {"hifigan_latency": hifigan_measures[-1]})

                if args.output is not None and reps == 1:
                    for i, audio in enumerate(hf_audios):
                        audio = audio[:mel_lens[i].item() * args.hop_length]
                        audio = audio.squeeze()
                        #audio = audio * args.max_wav_value
                        audio = audio.cpu().numpy().astype('int16')


                        #if args.fade_out:
                        #    fade_len = args.fade_out * args.hop_length
                        #    fade_w = torch.linspace(1.0, 0.0, fade_len)
                        #    audio[-fade_len:] *= fade_w.to(audio.device)
                        #audio = audio / torch.max(torch.abs(audio))
                        fname = f'{utt_id[i]}_audio_hifigan_{i}.wav' #b['output'][i] if 'output' in b else f'audio_{i}.wav'
                        audio_path = Path(args.output, fname)
                        write(audio_path, args.sampling_rate, audio)

            if mstft is not None:
                if args.output is not None and reps == 1:
                    for i00, mel_ in enumerate(mel):
                        if True:
                            ## with torch
                            spectrogram_to_wav_griffin_lim_sav( mel, mstft2,  args.output,  f'{utt_ids[j]}', args.sampling_rate, suffix='_audio_griffinlim_{i}.wav')


                        #audio= griffin_lim(mel, mstft)
                        #fname = f'{utt_id[i]}_audio_griffinlim_{i}.wav' #b['output'][i] if 'output' in b else f'audio_{i}.wav'
                        #audio_path = Path(args.output, fname)
                        #write(audio_path, args.sampling_rate, audio)

            if generator is not None and waveglow is not None:
                log(rep, {"latency": (gen_measures[-1] + waveglow_measures[-1])})

    log_enabled = True
    if generator is not None:
        gm = np.sort(np.asarray(gen_measures))
        rtf = all_samples / (all_utterances * gm.mean() * args.sampling_rate)
        log((), {"avg_fastpitch_letters/s": all_letters / gm.sum()})
        log((), {"avg_fastpitch_frames/s": all_frames / gm.sum()})
        log((), {"avg_fastpitch_latency": gm.mean()})
        log((), {"avg_fastpitch_RTF": rtf})
        log((), {"90%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.90) / 2) * gm.std()})
        log((), {"95%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.95) / 2) * gm.std()})
        log((), {"99%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.99) / 2) * gm.std()})
    if waveglow is not None:
        wm = np.sort(np.asarray(waveglow_measures))
        rtf = all_samples / (all_utterances * wm.mean() * args.sampling_rate)
        log((), {"avg_waveglow_samples/s": all_samples / wm.sum()})
        log((), {"avg_waveglow_latency": wm.mean()})
        log((), {"avg_waveglow_RTF": rtf})
        log((), {"90%_waveglow_latency": wm.mean() + norm.ppf((1.0 + 0.90) / 2) * wm.std()})
        log((), {"95%_waveglow_latency": wm.mean() + norm.ppf((1.0 + 0.95) / 2) * wm.std()})
        log((), {"99%_waveglow_latency": wm.mean() + norm.ppf((1.0 + 0.99) / 2) * wm.std()})
    if hifigan is not None:
        wm = np.sort(np.asarray(hifigan_measures))
        rtf = hf_all_samples / (hf_all_utterances * wm.mean() * args.sampling_rate)
        log((), {"avg_hifigan_samples/s": hf_all_samples / wm.sum()})
        log((), {"avg_hifigan_latency": wm.mean()})
        log((), {"avg_hifigan_RTF": rtf})
        log((), {"90%_hifigan_latency": wm.mean() + norm.ppf((1.0 + 0.90) / 2) * wm.std()})
        log((), {"95%_hifigan_latency": wm.mean() + norm.ppf((1.0 + 0.95) / 2) * wm.std()})
        log((), {"99%_hifigan_latency": wm.mean() + norm.ppf((1.0 + 0.99) / 2) * wm.std()})
    if generator is not None and waveglow is not None:
        m = gm + wm
        rtf = all_samples / (all_utterances * m.mean() * args.sampling_rate)
        log((), {"avg_samples/s": all_samples / m.sum()})
        log((), {"avg_letters/s": all_letters / m.sum()})
        log((), {"avg_latency": m.mean()})
        log((), {"avg_RTF": rtf})
        log((), {"90%_latency": m.mean() + norm.ppf((1.0 + 0.90) / 2) * m.std()})
        log((), {"95%_latency": m.mean() + norm.ppf((1.0 + 0.95) / 2) * m.std()})
        log((), {"99%_latency": m.mean() + norm.ppf((1.0 + 0.99) / 2) * m.std()})
    DLLogger.flush()


if __name__ == '__main__':
    main()
