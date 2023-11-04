# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import copy
import glob
import os
import re
import time
import warnings
from collections import defaultdict, OrderedDict

import json
import os, sys
import random


import traceback
from pathlib import Path
#from utils.samplers import RandomImbalancedSampler, PerfectBatchSampler,BalancedBatchSampler
#sys.path.append(os.path.dirname(__file__))
#sys.path.append(os.path.join(os.path.dirname(__file__), "fastpitch"))
#sys.path.append('.')
#sys.path.append(os.getcwd())

#https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/
#and
#https://github.com/Tomiinek/Multilingual_Text_to_Speech/

try:
    import nvidia_dlprof_pytorch_nvtx as pyprof
except ModuleNotFoundError:
    try:
        import pyprof
    except ModuleNotFoundError:
        warnings.warn('PyProf is unavailable')

import numpy as np
import torch
import torch.cuda.profiler as profiler
import torch.distributed as dist
if os.path.exists('/home/roseline/benit/apex/'):
    sys.path.insert(0, '/home/roseline/benit/apex/')
if os.path.exists('/home/roseline/apex/'):
    sys.path.insert(0, '/home/roseline/apex/')

from apex.optimizers import FusedAdam, FusedLAMB
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import common.tb_dllogger as logger
import models
from common.text import cmudict
from common.utils import prepare_tmp
from fastpitch.attn_loss_function import AttentionBinarizationLoss
from fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
from fastpitch.data_function_me import TTSDatasetMeMulti, TTSCollateMeMulti, batch_to_gpu_me
from fastpitch.data_function_me_char import TTSDatasetMeMultiChar, TTSCollateMeMultiChar, batch_to_gpu_me_char, batch_to_gpu_me_char_with_wav
from fastpitch.loss_function import FastPitchLoss, FastPitchWithWavLoss
from torchsummary import summary

from coqui_ai.tacotron.losses import TacotronLoss
from coqui_ai.mdnlosses import AlignTTSLoss
from  coqui_ai.mixer_tts.mixer_tts import MixerLoss
from coqui_ai.vits.vits_losses import VitsLoss
from coqui_ai.fastspeech.fastspeech2_submodules import FastSpeechLoss
import common.layers as layers
from common.audio_processing import griffin_lim, spectrogram_to_wav_griffin_lim_sav
from common.stft import STFT
import common.tf_griffin_lim as tfgriff


#def parse_args(parser):
#    return parser

#TYPE_ME="fastpitch" #"tensorflowtts"

def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)


def init_distributed(args, world_size, rank):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing distributed training")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(backend=('nccl' if args.cuda else 'gloo'),
                            init_method='env://')
    print("Done initializing distributed training")


def last_checkpoint(output, model_name):

    def corrupted(fpath):
        try:
            torch.load(fpath, map_location='cpu')
            return False
        except:
            warnings.warn(f'Cannot load {fpath}')
            return True

    saved = sorted(
        glob.glob(f'{output}/{model_name}_checkpoint_*.pt'),
        key=lambda f: int(re.search('_(\d+).pt', f).group(1)))

    if len(saved) >= 1 and not corrupted(saved[-1]):
        return saved[-1]
    elif len(saved) >= 2:
        return saved[-2]
    else:
        return None


def maybe_save_checkpoint(args, model, ema_model, optimizer, scaler, epoch,
                          total_iter, config, final_checkpoint=False):
    if args.local_rank != 0:
        return

    intermediate = (args.epochs_per_checkpoint > 0
                    and epoch % args.epochs_per_checkpoint == 0)

    if not intermediate and epoch < args.epochs and not final_checkpoint:
        return

    fpath = os.path.join(args.output, f"{model.name}_checkpoint_{epoch}.pt")
    print(f"Saving model and optimizer state at epoch {epoch} to {fpath}")
    ema_dict = None if ema_model is None else ema_model.state_dict()
    checkpoint = {'epoch': epoch,
                  'iteration': total_iter,
                  'config': config,
                  'state_dict': model.state_dict(),
                  'ema_state_dict': ema_dict,
                  'optimizer': optimizer.state_dict()}
    if args.amp:
        checkpoint['scaler'] = scaler.state_dict()
    torch.save(checkpoint, fpath)


def load_checkpoint000(args, model, ema_model, optimizer, scaler, epoch,
                    total_iter, config, filepath):
    if args.local_rank == 0:
        print(f'Loading model and optimizer state from {filepath}')
    checkpoint = torch.load(filepath, map_location='cpu')
    epoch[0] = checkpoint['epoch'] + 1
    total_iter[0] = checkpoint['iteration']

    sd = {k.replace('module.', ''): v
          for k, v in checkpoint['state_dict'].items()}
    getattr(model, 'module', model).load_state_dict(sd)
    #load_partial_state_dict(model, sd,  search_replace_str="module.", replace_by_str="",
    #                        search_replace_str2=None, replace_by_str2=None)
    optimizer.load_state_dict(checkpoint['optimizer'])

    if args.amp:
        scaler.load_state_dict(checkpoint['scaler'])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])

def load_partial_state_dict000(model, ckpt_state_dict, search_replace_str="module.", replace_by_str="",
                            search_replace_str2=None, replace_by_str2=None):
    model_dict = model.state_dict()
    nbr_found=0
    nbr_not_found=0
    for k in ckpt_state_dict.keys():
        if k in model_dict and model_dict[k].shape==ckpt_state_dict[k].shape:
            pname = k
            pval = ckpt_state_dict[k]
            print("load_finetune_checkpoint loading ",k)
            model_dict[pname] = pval.clone().to(model_dict[pname].device)
            nbr_found=nbr_found+1
        elif k.replace(search_replace_str,replace_by_str) in model_dict \
                and model_dict[k.replace(search_replace_str,replace_by_str)].shape==ckpt_state_dict[k].shape:
            #pname = k
            pval = ckpt_state_dict[k]
            print("load_finetune_checkpoint loading ",k," in ", k.replace(search_replace_str,replace_by_str))
            model_dict[k.replace(search_replace_str,replace_by_str)] = pval.clone().to(model_dict[k.replace(search_replace_str,replace_by_str)].device)
            nbr_found=nbr_found+1
        elif search_replace_str2 is not None and \
            k.replace(search_replace_str2,replace_by_str2) in model_dict \
                and model_dict[k.replace(search_replace_str2,replace_by_str2)].shape==ckpt_state_dict[k].shape:
            #pname = k
            pval = ckpt_state_dict[k]
            print("load_finetune_checkpoint loading ",k," in ", k.replace(search_replace_str2,replace_by_str2))
            model_dict[k.replace(search_replace_str2,replace_by_str2)] = pval.clone().to(model_dict[k.replace(search_replace_str2,replace_by_str2)].device)
            nbr_found=nbr_found+1
        else:
            nbr_not_found=nbr_not_found+1
        #self.load_state_dict(model_dict)
    #model.load_state_dict(model_dict)
    getattr(model, 'module', model).load_state_dict(model_dict)
    print(f'Loading partial model and optimizer state from file with found {nbr_found} values vs not found {nbr_not_found} keys')


def load_checkpoint(args, model, ema_model, optimizer, scaler, epoch,
                    total_iter, config, filepath, filepathsub=None):
    if args.local_rank == 0:
        print(f'Loading model and optimizer state from {filepath}')
    checkpoint = torch.load(filepath, map_location='cpu')
    epoch[0] = checkpoint['epoch'] + 1
    total_iter[0] = checkpoint['iteration']

    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        try:
            getattr(model, 'module', model).load_state_dict(checkpoint['state_dict'])
        except:
            try:
                sd = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
                getattr(model, 'module', model).load_state_dict(sd)
            except:
                traceback.print_exc()
                traceback.print_stack()
                try:
                    #load_partial_state_dict(model=getattr(model, 'module', model), ckpt_state_dict=checkpoint['state_dict'], search_replace_str="module.",
                    load_partial_state_dict(model=model, ckpt_state_dict=checkpoint['state_dict'], search_replace_str="module.",
                                            replace_by_str="", search_replace_str2=None, replace_by_str2=None, str_key_prefixe=None, diff_tolerable=10)
                except:
                    traceback.print_exc()
                    traceback.print_stack()
                    load_partial_state_dict(model=getattr(model, 'module', model), ckpt_state_dict=checkpoint['state_dict'],
                                            search_replace_str="module.",
                                            replace_by_str="", search_replace_str2=None, replace_by_str2=None,
                                            str_key_prefixe=None, diff_tolerable=10)

    if filepathsub is not None and os.path.exists(filepathsub):
        checkpointsub= torch.load(filepathsub, map_location='cpu')
        print("### Loading checkpoint sub ",filepathsub)
        load_partial_state_dict(model=getattr(model, 'module', model), ckpt_state_dict=checkpointsub['state_dict'],
                                search_replace_str="module.",
                                replace_by_str="", search_replace_str2=None, replace_by_str2=None,
                                str_key_prefixe=None, diff_tolerable=None)
        del checkpointsub
    #load_partial_state_dict(model, sd,  search_replace_str="module.", replace_by_str="",
    #                        search_replace_str2=None, replace_by_str2=None)
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        traceback.print_exc()
        traceback.print_stack()

    if args.amp:
        scaler.load_state_dict(checkpoint['scaler'])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])

def load_partial_state_dict(model, ckpt_state_dict, search_replace_str="module.", replace_by_str="",
                            search_replace_str2=None, replace_by_str2=None, str_key_prefixe=None, diff_tolerable=10):
    model_dict = model.state_dict()
    print("loading partial checkpoint from state dict loading ")
    nbr_found=0
    nbr_not_found=0
    n_target_keys=len(model_dict.keys())
    for k in ckpt_state_dict.keys():
        if k in model_dict and model_dict[k].shape==ckpt_state_dict[k].shape:
            pname = k
            pval = ckpt_state_dict[k]
            print("load_finetune_checkpoint loading ",k,"\n")
            model_dict[pname] = pval.clone().to(model_dict[pname].device)
            nbr_found=nbr_found+1
        elif k.replace(search_replace_str,replace_by_str) in model_dict \
                and model_dict[k.replace(search_replace_str,replace_by_str)].shape==ckpt_state_dict[k].shape:
            #pname = k
            pval = ckpt_state_dict[k]
            print("load_finetune_checkpoint loading ",k," in ", k.replace(search_replace_str,replace_by_str),"\n")
            model_dict[k.replace(search_replace_str,replace_by_str)] = pval.clone().to(model_dict[k.replace(search_replace_str,replace_by_str)].device)
            nbr_found=nbr_found+1
        elif search_replace_str2 is not None and \
            k.replace(search_replace_str2,replace_by_str2) in model_dict \
                and model_dict[k.replace(search_replace_str2,replace_by_str2)].shape==ckpt_state_dict[k].shape:
            #pname = k
            pval = ckpt_state_dict[k]
            print("load_finetune_checkpoint loading ",k," in ", k.replace(search_replace_str2,replace_by_str2),"\n")
            model_dict[k.replace(search_replace_str2,replace_by_str2)] = pval.clone().to(model_dict[k.replace(search_replace_str2,replace_by_str2)].device)
            nbr_found=nbr_found+1
        elif str_key_prefixe is not None and str_key_prefixe + k in model_dict \
                and model_dict[str_key_prefixe + k].shape == ckpt_state_dict[k].shape:
            # pname = k
            pval = ckpt_state_dict[k]
            print("load_finetune_checkpoint loading ", k, " in ",
                  str_key_prefixe + k,"\n")
            model_dict[str_key_prefixe + k] = pval.clone().to(
                model_dict[str_key_prefixe + k].device)
            nbr_found = nbr_found + 1
            found = True
            break
        else:
            nbr_not_found=nbr_not_found+1
        #self.load_state_dict(model_dict)
    #model.load_state_dict(model_dict)
    if diff_tolerable is not None:
        assert abs(n_target_keys-nbr_found)<=diff_tolerable, f"Unmatched number of keys for state dict loading target model={n_target_keys} vs source ckp {nbr_found}\n"
    getattr(model, 'module', model).load_state_dict(model_dict)
    print(f'Loading partial model and optimizer state from file with found {nbr_found} values vs not found {nbr_not_found} keys')


def load_partial_state_dict_from_list(model, ckpt_state_dict_list,
                            search_replace_str_list=["module."], replace_by_str_list=[""],
                            str_prefix_list=None):
    model_dict = model.state_dict()
    nbr_found=0
    nbr_not_found=0
    assert isinstance(ckpt_state_dict_list, list)," statedict must be a list"
    assert isinstance(search_replace_str_list, list) or search_replace_str_list is None," statedict must be a list"
    assert isinstance(replace_by_str_list, list) or replace_by_str_list is None," statedict must be a list"
    assert isinstance(str_prefix_list, list) or str_prefix_list is None," str_prefix_list must be a list"
    assert len(search_replace_str_list)==len(replace_by_str_list), "search string list and replaced string list must have same size"

    for ckpt_state_dict in  ckpt_state_dict_list:
        for k in ckpt_state_dict.keys():
            if k in model_dict and model_dict[k].shape==ckpt_state_dict[k].shape:
                pname = k
                pval = ckpt_state_dict[k]
                print("load_finetune_checkpoint loading ",k)
                model_dict[pname] = pval.clone().to(model_dict[pname].device)
                nbr_found=nbr_found+1
            else:
                found=False
                for i_replace, searchstr in enumerate( search_replace_str_list):
                    search_replace_str= search_replace_str_list[i_replace]
                    replace_by_str=  replace_by_str_list[i_replace]
                    if k.replace(search_replace_str,replace_by_str) in model_dict \
                        and model_dict[k.replace(search_replace_str,replace_by_str)].shape==ckpt_state_dict[k].shape:
                        #pname = k
                        pval = ckpt_state_dict[k]
                        print("load_finetune_checkpoint loading ",k," in ", k.replace(search_replace_str,replace_by_str))
                        model_dict[k.replace(search_replace_str,replace_by_str)] = pval.clone().to(model_dict[k.replace(search_replace_str,replace_by_str)].device)
                        nbr_found=nbr_found+1
                        found= True
                        break
                if not found:
                    for i_prefix, key_prefixe in enumerate(str_prefix_list):
                        str_key_prefixe = str_prefix_list[i_prefix]
                        if str_key_prefixe+k in model_dict \
                                and model_dict[str_key_prefixe+k].shape == ckpt_state_dict[k].shape:
                            # pname = k
                            pval = ckpt_state_dict[k]
                            print("load_finetune_checkpoint loading ", k, " in ",
                                  str_key_prefixe+k)
                            model_dict[str_key_prefixe+k] = pval.clone().to(
                                model_dict[str_key_prefixe+k].device)
                            nbr_found = nbr_found + 1
                            found = True
                            break
            # elif search_replace_str2 is not None and \
            #     k.replace(search_replace_str2,replace_by_str2) in model_dict \
            #         and model_dict[k.replace(search_replace_str2,replace_by_str2)].shape==ckpt_state_dict[k].shape:
            #     #pname = k
            #     pval = ckpt_state_dict[k]
            #     print("load_finetune_checkpoint loading ",k," in ", k.replace(search_replace_str2,replace_by_str2))
            #     model_dict[k.replace(search_replace_str2,replace_by_str2)] = pval.clone().to(model_dict[k.replace(search_replace_str2,replace_by_str2)].device)
            #     nbr_found=nbr_found+1
                if not found:
                    nbr_not_found=nbr_not_found+1
            #self.load_state_dict(model_dict)
    #model.load_state_dict(model_dict)
    getattr(model, 'module', model).load_state_dict(model_dict)
    print(f'Loading partial model and optimizer state from file with found {nbr_found} values vs not found {nbr_not_found} keys')


def validate(model, epoch, total_iter, criterion, valset, batch_size,
             collate_fn, distributed_run, batch_to_gpu, ema=False):
    """Handles all the validation scoring and printing"""
    was_training = model.training
    model.eval()

    tik = time.perf_counter()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=1, shuffle=False,
                                sampler=val_sampler,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn)
        val_meta = defaultdict(float)
        val_num_frames = 0
        for i, batch in enumerate(val_loader):
            x, y, num_frames = batch_to_gpu(batch)
            y_pred = model(x)
            loss, meta = criterion(y_pred, y, is_training=False, meta_agg='sum')

            if distributed_run:
                for k, v in meta.items():
                    val_meta[k] += reduce_tensor(v, 1)
                val_num_frames += reduce_tensor(num_frames.data, 1).item()
            else:
                for k, v in meta.items():
                    val_meta[k] += v
                val_num_frames = num_frames.item()

        val_meta = {k: v / len(valset) for k, v in val_meta.items()}

    val_meta['took'] = time.perf_counter() - tik

    logger.log((epoch,) if epoch is not None else (),
               tb_total_steps=total_iter,
               subset='val_ema' if ema else 'val',
               data=OrderedDict([
                   ('loss', val_meta['loss'].item()),
                   ('mel_loss', val_meta['mel_loss'].item()),
                   ('frames/s', num_frames.item() / val_meta['took']),
                   ('took', val_meta['took'])]),
               )

    if was_training:
        model.train()
    return val_meta

def validate_me(model, epoch, total_iter, criterion, valset, batch_size,
             collate_fn, distributed_run, batch_to_gpu_me_char_with_wav,
                ema=False, use_tones=False, use_wav=False,
                obj_griffin_lim=None, outwavdir=None, sampling_rate=16000):
    """Handles all the validation scoring and printing"""
    was_training = model.training
    model.eval()
    id_wav_gen= random.randint(0, 20)#len(valset)//(2*batch_size))

    tik = time.perf_counter()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=1, shuffle=False,
                                sampler=val_sampler, drop_last=True,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn)
        val_meta = defaultdict(float)
        val_num_frames = 0
        for i, batch in enumerate(val_loader):
            try:
                x, y, num_frames = batch_to_gpu_me_char_with_wav(batch, use_tones=use_tones, use_wav=use_wav)
                utt_id= x[10]  if  use_tones else x[9]
                y_pred = model(x)
                loss, meta = criterion(y_pred, y, is_training=False, meta_agg='sum')
                if i==id_wav_gen :
                    print("##### generating wav ")
                    if obj_griffin_lim is not None and outwavdir is not None:
                        index_in_batch= 0
                        test_wav_fname= f'test_{epoch}_batch_{i}_{utt_id[index_in_batch]}'  if utt_id is not None else f'test_epoch_{epoch}_batch_{i}'
                        spectrogram_to_wav_griffin_lim_sav(y_pred[index_in_batch], obj_griffin_lim, outwavdir, test_wav_fname , sampling_rate,
                                                           suffix=f'_val_gf.wav')

                if distributed_run:
                    for k, v in meta.items():
                        val_meta[k] += reduce_tensor(v, 1)
                    val_num_frames += reduce_tensor(num_frames.data, 1).item()
                else:
                    for k, v in meta.items():
                        val_meta[k] += v
                    val_num_frames = num_frames.item()
            except:
                continue

        val_meta = {k: v / len(valset) for k, v in val_meta.items()}

    val_meta['took'] = time.perf_counter() - tik

    logger.log((epoch,) if epoch is not None else (),
               tb_total_steps=total_iter,
               subset='val_ema' if ema else 'val',
               data=OrderedDict([
                   ('loss', val_meta['loss'].item()),
                   ('mel_loss', val_meta['mel_loss'].item()),
                   ('frames/s', num_frames.item() / val_meta['took']),
                   ('took', val_meta['took'])]),
               )

    if was_training:
        model.train()
    return val_meta


def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate * scale


def apply_ema_decay(model, ema_model, decay):
    if not decay:
        return
    st = model.state_dict()
    add_module = hasattr(model, 'module') and not hasattr(ema_model, 'module')
    for k, v in ema_model.state_dict().items():
        if add_module and not k.startswith('module.'):
            k = 'module.' + k
        v.copy_(decay * v + (1 - decay) * st[k])


def init_multi_tensor_ema(model, ema_model):
    model_weights = list(model.state_dict().values())
    ema_model_weights = list(ema_model.state_dict().values())
    ema_overflow_buf = torch.cuda.IntTensor([0])
    return model_weights, ema_model_weights, ema_overflow_buf


def apply_multi_tensor_ema(decay, model_weights, ema_weights, overflow_buf):
    import amp_C
    amp_C.multi_tensor_axpby(
        65536, overflow_buf, [ema_weights, model_weights, ema_weights],
        decay, 1-decay, -1)


def main():
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Training',
                                     allow_abbrev=False)

    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save checkpoints')
    #parser.add_argument('-d', '--dataset-path', type=str, default='./',   help='Path to dataset')
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
        "--train_meta", default="dump/meta_train.csv", type=str,
    )
    myfonfrparam.add_argument(
        "--valid_meta", default="dump/meta_valid.csv", type=str,
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
    myfonfrparam.add_argument(
        "--variant", default="", type=str,
    )

    myfonfrparam.add_argument('--train-type', default='fastpitch',
                      choices=['fastpitch','tensorflowtts', "create"],
                      help='using fastspeech data or tensorflowtts data')

    myfonfrparam.add_argument('--format', default='npy',
                      choices=['npy','pt'],
                      help='using fastspeech data npy or tensorflowtts data npy')

    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--pyprof', action='store_true',
                        help='Enable pyprof profiling')

    train = parser.add_argument_group('training setup')
    train.add_argument('--epochs', type=int, required=True,
                       help='Number of total epochs to run')
    train.add_argument('--epochs-per-checkpoint', type=int, default=1,
                       help='Number of epochs per checkpoint')
    train.add_argument('--checkpoint-path', type=str, default=None,
                       help='Checkpoint path to resume training')
    train.add_argument('--resume', action='store_true',
                       help='Resume training from the last checkpoint')
    train.add_argument('--seed', type=int, default=1234,
                       help='Seed for PyTorch random number generators')
    train.add_argument('--amp', action='store_true',
                       help='Enable AMP')
    train.add_argument('--cuda', action='store_true',
                       help='Run on GPU using CUDA')
    train.add_argument('--cudnn-benchmark', action='store_true',
                       help='Enable cudnn benchmark mode')
    train.add_argument('--ema-decay', type=float, default=0,
                       help='Discounting factor for training weights EMA')
    train.add_argument('--grad-accumulation', type=int, default=1,
                       help='Training steps to accumulate gradients for')
    train.add_argument('--kl-loss-start-epoch', type=int, default=250,
                       help='Start adding the hard attention loss term')
    train.add_argument('--kl-loss-warmup-epochs', type=int, default=100,
                       help='Gradually increase the hard attention loss term')
    train.add_argument('--kl-loss-weight', type=float, default=1.0,
                       help='Gradually increase the hard attention loss term')


    opt = parser.add_argument_group('optimization setup')
    opt.add_argument('--optimizer', type=str, default='lamb',
                     help='Optimization algorithm')
    opt.add_argument('-lr', '--learning-rate', type=float, default=1e-3, 
                     help='Learing rate')
    opt.add_argument('--weight-decay', default=1e-6, type=float,
                     help='Weight decay')
    opt.add_argument('--grad-clip-thresh', default=1000.0, type=float,
                     help='Clip threshold for gradients')
    opt.add_argument('-bs', '--batch-size', type=int, required=True,
                     help='Batch size per GPU')
    opt.add_argument('--warmup-steps', type=int, default=100000,
                     help='Number of steps for lr warmup')
    opt.add_argument('--dur-predictor-loss-scale', type=float,
                     default=1.0, help='Rescale duration predictor loss')
    opt.add_argument('--pitch-predictor-loss-scale', type=float,
                     default=1.0, help='Rescale pitch predictor loss')
    opt.add_argument('--attn-loss-scale', type=float,
                     default=1.0, help='Rescale alignment loss')
    opt.add_argument('--use-sdtw', action='store_true', help='Rescale alignment loss')

    #opt.add_argument('--classifier-loss-scale', type=float,
    #                 default=0.5, help='Rescale alignment loss')
    ##opt.add_argument('--reversal-classifier', action='store_true',
    ##                  help='use reversal classifier')
    #opt.add_argument('--reversal-classifier-type', type=str,
    #                 default="reversal", help='reversal classifier type')
    #opt.add_argument('--reversal-classifier-w', type=float,
    #                 default=1.0, help='Rescale alignment loss')


    data = parser.add_argument_group('dataset parameters')
    #data.add_argument('--training-files', type=str, nargs='*', required=True,
    #                  help='Paths to training filelists.')
    #data.add_argument('--validation-files', type=str, nargs='*',
    #                  required=True, help='Paths to validation filelists')
    data.add_argument(
        "--train-dir",
        default="dump/train",
        type=str,
        help="directory including training data. ",
    )
    data.add_argument(
        "--dev-dir",
        default="dump/valid",
        type=str,
        help="directory including development data. ",
    )
    data.add_argument('--text-cleaners', nargs='*',
                      default=['english_cleaners'], type=str,
                      help='Type of text cleaners for input text')
    data.add_argument('--symbol-set', type=str, default='english_basic',
                      help='Define symbol set for input text')
    data.add_argument('--p-arpabet', type=float, default=0.0,
                      help='Probability of using arpabets instead of graphemes '
                           'for each word; set 0 for pure grapheme training')
    data.add_argument('--heteronyms-path', type=str, default='cmudict/heteronyms',
                      help='Path to the list of heteronyms')
    data.add_argument('--cmudict-path', type=str, default='cmudict/cmudict-0.7b',
                      help='Path to the pronouncing dictionary')
    data.add_argument('--prepend-space-to-text', action='store_true',
                      help='Capture leading silence with a space token')
    data.add_argument('--append-space-to-text', action='store_true',
                      help='Capture trailing silence with a space token')
    data.add_argument('--train-wav', action='store_true',
                      help='add wav and spectrogram in dataset output')
    data.add_argument('--verbose', action='store_true',
                      help='mode verbose')


    data.add_argument('--no-load-pickle', action='store_true', # default=False,
                      help='add wav and spectrogram in dataset output')

    cond = parser.add_argument_group('data for conditioning')
    #cond.add_argument('--n-speakers', type=int, default=1,
    #                  help='Number of speakers in the dataset. '
    #                       'n_speakers > 1 enables speaker embeddings')
    #cond.add_argument('--n-languages', type=int, default=1,
    #                  help='Number of languages in the dataset. '
    #                       'n_languages > 1 enables language embeddings')
    #cond.add_argument('--load-pitch-from-disk', action='store_true',
    #                  help='Use pitch cached on disk with prepare_dataset.py')
    cond.add_argument('--n-speakers-add', type=int, default=10,
                      help='Number of speakers to add to ensure availability to add speakers easily without modifing code. '
                           'n_speakers > 1 enables speaker embeddings')
    cond.add_argument('--n-languages-add', type=int, default=3,
                      help='Number of languages to add to ensure availability to add languages easily without modifing code. '
                           'n_languages > 1 enables speaker embeddings')

    cond.add_argument('--pitch-online-method', default='pyin',
                      choices=['pyworld','pyin'],
                      help='Calculate pitch on the fly during training')
    cond.add_argument('--pitch-online-dir', type=str, default=None,
                      help='A directory for storing pitch calculated on-line')
    cond.add_argument('--pitch-mean', type=float, default=214.72203,
                      help='Normalization value for pitch')
    cond.add_argument('--pitch-std', type=float, default=65.72038,
                      help='Normalization value for pitch')
    #cond.add_argument('--load-mel-from-disk', action='store_true',
    #                  help='Use mel-spectrograms cache on the disk')  # XXX

    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--max-wav-value', default=32768.0, type=float,
                       help='Maximum audiowave value')
    audio.add_argument('--sampling-rate', default=16000, type=int,
                       help='Sampling rate')
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

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0),
                      help='Rank of the process for multiproc; do not set manually')
    dist.add_argument('--world_size', type=int, default=os.getenv('WORLD_SIZE', 1),
                      help='Number of processes for multiproc; do not set manually')

    #parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    #args.world_size= 1

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
        #print("££££££ test ", args.filter_remove_lang_ids, len(args.filter_remove_lang_ids))
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
        #mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"


        duration_query =  "*-durations.npy"
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
        

    #if args.p_arpabet > 0.0:
    #    cmudict.initialize(args.cmudict_path, keep_ambiguous=True)

    distributed_run = args.world_size > 1 

    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)

    if args.local_rank == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    log_fpath = args.log_file or os.path.join(args.output, 'nvlog.json')
    tb_subsets = ['train', 'val']
    if args.ema_decay > 0.0:
        tb_subsets.append('val_ema')

    logger.init(log_fpath, args.output, enabled=(args.local_rank == 0),
                tb_subsets=tb_subsets)
    logger.parameters(vars(args), tb_subset='train')

    if args.variant is not None and args.variant!="":
        if args.variant.lower()=="fastspeechmodel":
            parser = models.parse_model_args('FastSpeechModel', parser)
        elif args.variant.lower()=="fastpitchmultiwithaligner":
            parser = models.parse_model_args('FastPitchMultiWithAligner', parser)
        elif args.variant.lower()=="fastpitchmultiwithmixer":
            parser = models.parse_model_args('FastPitchMultiWithMixer', parser)
        elif args.variant.lower()=="fastpitchmultiwithvitsaligner":
            parser = models.parse_model_args('FastPitchMultiWithVITSAligner', parser)
        elif args.variant.lower()=="tacotron2":
            parser = models.parse_model_args('Tacotron2', parser)
        else:
            raise ValueError(f'not yet implemented')
    elif args.generated:
        parser = models.parse_model_args('FastPitchMultiGenerated', parser)
    elif args.tones:
        raise ValueError(f'not yet implemented')
    elif args.toneseparated:
        raise ValueError(f'not yet implemented')
    elif args.train_wav:
        parser = models.parse_model_args('FastPitchMultiWav', parser)
    else:
        parser = models.parse_model_args('FastPitchMulti', parser)

    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if distributed_run:
        init_distributed(args, args.world_size, args.local_rank)

    device = torch.device('cuda' if args.cuda else 'cpu')
    

    speakers_map = None
    languages_map = None

    with open(args.dataset_mapping) as f:
        dataset_mapping = json.load(f)
        speakers_map = dataset_mapping["speakers_map"]
        languages_map = dataset_mapping["languages_map"]

    args.n_speakers=n_speakers=len(speakers_map)+ args.n_speakers_add#10
    args.n_languages=n_languages= len(languages_map)+args.n_languages_add#3
    args.n_mel_channels=args.nmel_channels

    if args.variant is not None and args.variant!="":
        if args.variant.lower()=="fastspeechmodel":
            model_config = models.get_model_config('FastSpeechModel', args)
        elif args.variant.lower()=="fastpitchmultiwithaligner":
            model_config = models.get_model_config('FastPitchMultiWithAligner', args)
        elif args.variant.lower()=="fastpitchmultiwithmixer":
            model_config = models.get_model_config('FastPitchMultiWithMixer', args)
        elif args.variant.lower()=="fastpitchmultiwithvitsaligner":
            model_config = models.get_model_config('FastPitchMultiWithVITSAligner', args)
        elif args.variant.lower()=="tacotron2":
            model_config = models.get_model_config('Tacotron2', args)
        else:
            raise ValueError(f'not yet implemented')
    elif args.generated:
        model_config = models.get_model_config('FastPitchMultiGenerated', args)
    elif args.tones:
        raise ValueError(f'not yet implemented')
    elif args.toneseparated:
        raise ValueError(f'not yet implemented')
    elif args.train_wav:
        model_config = models.get_model_config('FastPitchMultiWav', args)
    else:
        model_config = models.get_model_config('FastPitchMulti', args)


    if args.variant is not None and args.variant!="":
        if args.variant.lower()=="fastspeechmodel":
            model = models.get_model('FastSpeechModel',  model_config, device)
        elif args.variant.lower()=="fastpitchmultiwithaligner":
            model = models.get_model('FastPitchMultiWithAligner', model_config, device)
        elif args.variant.lower()=="fastpitchmultiwithmixer":
            model = models.get_model('FastPitchMultiWithMixer', model_config, device)
        elif args.variant.lower()=="fastpitchmultiwithvitsaligner":
            model = models.get_model('FastPitchMultiWithVITSAligner', model_config, device)
        elif args.variant.lower()=="tacotron2":
            model = models.get_model('Tacotron2', model_config, device)
        else:
            raise ValueError(f'not yet implemented')
    elif args.generated:
        model = models.get_model('FastPitchMultiGenerated', model_config, device)
    elif args.tones:
        raise ValueError(f'not yet implemented')
    elif args.toneseparated:
        raise ValueError(f'not yet implemented')
    elif args.train_wav:
        model = models.get_model('FastPitchMultiWav', model_config, device)
    else:
        model = models.get_model('FastPitchMulti', model_config, device)



    print("model summary ")
    print(model)
    #summary(model, (1, 27, 512))

    convert_ipa=False
    if args.convert_ipa:
        convert_ipa=True
    if args.use_ipa_phone:
        convert_ipa=False


    attention_kl_loss = AttentionBinarizationLoss()

    # Store pitch mean/std as params to translate from Hz during inference
    if hasattr(model, "pitch_mean"):
        model.pitch_mean[0] = args.pitch_mean
        model.pitch_std[0] = args.pitch_std

    kw = dict(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9,
              weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        optimizer = FusedAdam(model.parameters(), **kw)
    elif args.optimizer == 'lamb':
        optimizer = FusedLAMB(model.parameters(), **kw)
    else:
        raise ValueError

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model)
    else:
        ema_model = None

    if distributed_run:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)

    if args.pyprof:
        pyprof.init(enable_function_stack=True)

    start_epoch = [1]
    start_iter = [0]

    assert args.checkpoint_path is None or args.resume is False, (
        "Specify a single checkpoint source")
    if args.checkpoint_path is not None:
        ch_fpath = args.checkpoint_path
    elif args.resume:
        ch_fpath = last_checkpoint(args.output, model.name)
    else:
        ch_fpath = None

    if ch_fpath is not None:
        load_checkpoint(args, model, ema_model, optimizer, scaler,
                        start_epoch, start_iter, model_config, ch_fpath)

    start_epoch = start_epoch[0]
    total_iter = start_iter[0]

    if args.variant is not None and args.variant!="":
        if args.variant.lower()=="fastspeechmodel":
            criterion = FastSpeechLoss(energy_predictor_loss_scale=0.1,
                dur_predictor_loss_scale=args.dur_predictor_loss_scale, attn_loss_scale=args.attn_loss_scale,
                pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
                classifier_loss_scale=args.classifier_loss_scale, 
                reversal_classifier=bool(args.use_reversal_classifier), #args.reversal_classifier, 
                reversal_classifier_type=args.reversal_classifier_type, 
                reversal_classifier_w=args.reversal_classifier_w, n_mel_channels=args.nmel_channels,
                )
        elif args.variant.lower()=="fastpitchmultiwithaligner":
            criterion = AlignTTSLoss(energy_predictor_loss_scale=0.1,
                dur_predictor_loss_scale=args.dur_predictor_loss_scale,
                pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
                 classifier_loss_scale=args.classifier_loss_scale, 
                reversal_classifier=bool(args.use_reversal_classifier), #args.reversal_classifier, 
                reversal_classifier_type=args.reversal_classifier_type, 
                reversal_classifier_w=args.reversal_classifier_w, n_mel_channels=args.nmel_channels,
                ssim_alpha = 1.0, dur_loss_alpha= 1.0, spec_loss_alpha= 1.0, mdn_alpha = 1.0)
        elif args.variant.lower()=="fastpitchmultiwithmixer":
            criterion = MixerLoss(
                dur_predictor_loss_scale=args.dur_predictor_loss_scale,
                pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
                attn_loss_scale=args.attn_loss_scale, classifier_loss_scale=args.classifier_loss_scale, 
                reversal_classifier=bool(args.use_reversal_classifier), #args.reversal_classifier, 
                reversal_classifier_type=args.reversal_classifier_type, 
                reversal_classifier_w=args.reversal_classifier_w, n_mel_channels=args.nmel_channels, energy_predictor_loss_scale=0.1,  
                  bin_loss_scale=0.0, mel_loss_scale=1.0,  bin_loss_start_ratio= 0.2, bin_loss_warmup_epochs=100, add_bin_loss=False)
        elif args.variant.lower()=="fastpitchmultiwithvitsaligner":
            criterion = VitsLoss(energy_predictor_loss_scale=0.1, attn_loss_scale=1.0,
                dur_predictor_loss_scale=args.dur_predictor_loss_scale,
                pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
                 classifier_loss_scale=args.classifier_loss_scale, 
                reversal_classifier=bool(args.use_reversal_classifier), #args.reversal_classifier, 
                reversal_classifier_type=args.reversal_classifier_type, 
                reversal_classifier_w=args.reversal_classifier_w, n_mel_channels=args.nmel_channels,
                )
        elif args.variant.lower()=="tacotron2":
            criterion = TacotronLoss( dur_predictor_loss_scale=args.dur_predictor_loss_scale,
                 pitch_predictor_loss_scale=args.pitch_predictor_loss_scale, attn_loss_scale=args.attn_loss_scale,
                 energy_predictor_loss_scale=0.1, classifier_loss_scale=args.classifier_loss_scale, 
                 reversal_classifier=bool(args.use_reversal_classifier), #args.reversal_classifier, , 
                 reversal_classifier_type=args.reversal_classifier_type, reversal_classifier_w=args.reversal_classifier_w, n_mel_channels=args.nmel_channels, stopnet_pos_weight= 10.0, ga_alpha= 5.0, decoder_diff_spec_alpha= 0.25, postnet_diff_spec_alpha= 0.25, decoder_loss_alpha= 0.25,  postnet_loss_alpha= 0.25, decoder_ssim_alpha= 0.25, postnet_ssim_alpha= 0.25, loss_masking= True, bidirectional_decoder= False, stopnet= True, double_decoder_consistency= False, outputs_per_step= model.config.r, config=model.config)
        else:
            raise ValueError(f'not yet implemented')
    elif args.generated:
        criterion = FastPitchLoss(
            dur_predictor_loss_scale=args.dur_predictor_loss_scale,
            pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
            attn_loss_scale=args.attn_loss_scale, classifier_loss_scale=args.classifier_loss_scale, 
            reversal_classifier=bool(args.use_reversal_classifier), #args.reversal_classifier, 
            reversal_classifier_type=args.reversal_classifier_type, 
            reversal_classifier_w=args.reversal_classifier_w, n_mel_channels=args.nmel_channels)

    elif args.tones:
        raise ValueError(f'not yet implemented')
    elif args.toneseparated:
        raise ValueError(f'not yet implemented')
    elif args.train_wav:
        criterion = FastPitchWithWavLoss(
            dur_predictor_loss_scale=args.dur_predictor_loss_scale,
            pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
            attn_loss_scale=args.attn_loss_scale, classifier_loss_scale=args.classifier_loss_scale,
            reversal_classifier=bool(args.use_reversal_classifier),  # args.reversal_classifier,
            reversal_classifier_type=args.reversal_classifier_type,
            reversal_classifier_w=args.reversal_classifier_w, n_mel_channels=args.nmel_channels,
            add_softdtw=args.use_sdtw #getattr(args, 'use_soft', False) or getattr(args, 'use_soft_tacotron', False)
        )
    else:
        criterion = FastPitchLoss(
            dur_predictor_loss_scale=args.dur_predictor_loss_scale,
            pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
            attn_loss_scale=args.attn_loss_scale, classifier_loss_scale=args.classifier_loss_scale, 
            reversal_classifier=bool(args.use_reversal_classifier), #args.reversal_classifier, 
            reversal_classifier_type=args.reversal_classifier_type, 
            reversal_classifier_w=args.reversal_classifier_w, n_mel_channels=args.nmel_channels)

    mstft2 = layers.TacotronSTFT(
        filter_length=args.filter_length, hop_length=args.hop_length, win_length=args.win_length,
        n_mel_channels=args.nmel_channels, sampling_rate=args.sampling_rate, mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax)
    # 4 workers are optimal on DGX-1 (from epoch 2 onwards)


    collate_fn = TTSCollateMeMultiChar()

    if args.local_rank == 0:
        prepare_tmp(args.pitch_online_dir)

    #print("££££££### test ", args.filter_remove_lang_ids, len(args.filter_remove_lang_ids))
    trainset = TTSDatasetMeMultiChar(
            loaded_mapper_path= args.dataset_mapping ,
            sentence_list_file= args.train_meta,
            root_dir=args.train_dir,
            wave_query=wave_query, #charactor_query=charactor_query, tones_query=tones_query,
            #speaker_query=speaker_query,
            #language_speaker_query=language_speaker_query,
            #language_query=language_query,
            #charactor_load_fn=np.load, language_load_fn=language_load_fn,
            #speaker_load_fn=speaker_load_fn,
            mel_query=mel_query, mel_load_fn=mel_load_fn,
            duration_query=duration_query, duration_load_fn=duration_load_fn,
            f0_query=f0_query, f0_load_fn=f0_load_fn, pitch_query=pitch_query,
            energy_query=energy_query, energy_load_fn=energy_load_fn,
            f0_stat=args.f0_stat,
            energy_stat=args.energy_stat,
            #mel_length_threshold=mel_length_threshold, max_mel_length_threshold=max_mel_length_threshold,
            speakers_map=speakers_map,
            languages_map=languages_map, convert_ipa=convert_ipa, use_tones=use_tones,
            filter_on_lang=args.filter_on_lang,
            filter_on_speaker=args.filter_on_speaker,
            filter_on_uttid=args.filter_on_uttid,
            filter_remove_lang_ids=l_filter_remove_lang_ids,        
            filter_remove_speaker_ids=l_filter_remove_speaker_ids,   
            filter_remove_uttid_ids=l_filter_remove_uttid_ids, 
            #load_language_array=args.load_language_array , load_speaker_array=args.load_speaker_array,
            #language_array_query=language_array_query,   speaker_array_query=speaker_array_query,

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
                pitch_online_method=args.pitch_online_method, #f0_method
                cache_file_prefix = 'train_',
                savepickle = not(args.no_load_pickle),
                use_priorgrad_wav=(args.train_wav and args.use_priorgrad),
                is_training_prior_grad=(args.train_wav and args.use_priorgrad),
                use_prior= True,
            ) #
    valset = TTSDatasetMeMultiChar(
            root_dir=args.dev_dir,

            loaded_mapper_path= args.dataset_mapping ,
            sentence_list_file= args.valid_meta,

            wave_query=wave_query, #charactor_query=charactor_query, tones_query=tones_query,
            #speaker_query=speaker_query,
            #language_speaker_query=language_speaker_query,
            #language_query=language_query,
            #charactor_load_fn=np.load, language_load_fn=language_load_fn,
            #speaker_load_fn=speaker_load_fn,
            mel_query=mel_query, mel_load_fn=mel_load_fn,
            duration_query=duration_query, duration_load_fn=duration_load_fn,
            f0_query=f0_query, f0_load_fn=f0_load_fn, pitch_query=pitch_query,
            energy_query=energy_query, energy_load_fn=energy_load_fn,
            f0_stat=args.f0_stat,
            energy_stat=args.energy_stat,
            #mel_length_threshold=mel_length_threshold, max_mel_length_threshold=max_mel_length_threshold,
            speakers_map=speakers_map,
            languages_map=languages_map, convert_ipa=convert_ipa, use_tones=use_tones,
            filter_on_lang=args.filter_on_lang,
            filter_on_speaker=args.filter_on_speaker,
            filter_on_uttid=args.filter_on_uttid,
            filter_remove_lang_ids=l_filter_remove_lang_ids,        
            filter_remove_speaker_ids=l_filter_remove_speaker_ids,   
            filter_remove_uttid_ids=l_filter_remove_uttid_ids, 
            #load_language_array=args.load_language_array , load_speaker_array=args.load_speaker_array,
            #language_array_query=language_array_query,   speaker_array_query=speaker_array_query,

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
                cache_file_prefix='valid_',
                savepickle=True,
                use_priorgrad_wav = (args.train_wav and args.use_priorgrad),
                is_training_prior_grad=False,
                use_prior= True,
             )#**vars(args)

    if  distributed_run:
        train_sampler, shuffle = DistributedSampler(trainset), False
    else:
        train_sampler, shuffle = None, True


    # 4 workers are optimal on DGX-1 (from epoch 2 onwards)
    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler, batch_size=args.batch_size,
                              pin_memory=True, persistent_workers=True,
                              drop_last=True, collate_fn=collate_fn)

    if args.ema_decay:
        mt_ema_params = init_multi_tensor_ema(model, ema_model)

    model.train()

    if args.pyprof:
        torch.autograd.profiler.emit_nvtx().__enter__()
        profiler.start()

    epoch_loss = []
    epoch_mel_loss = []
    epoch_lang_loss = []
    epoch_num_frames = []
    epoch_frames_per_sec = []
    epoch_time = []

    torch.cuda.synchronize()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()

        epoch_loss += [0.0]
        epoch_mel_loss += [0.0]
        epoch_lang_loss += [0.0]
        epoch_num_frames += [0]
        epoch_frames_per_sec += [0.0]

        if distributed_run:
            train_loader.sampler.set_epoch(epoch)

        accumulated_steps = 0
        iter_loss = 0
        iter_num_frames = 0
        iter_meta = {}
        iter_start_time = None

        epoch_iter = 0
        num_iters = len(train_loader) // args.grad_accumulation
        for batch in train_loader:
            #if batch[0].size(0)!=args.batch_size and args.generated:
            #    print("Batch size ",batch[0].size(0) ," under required ", args.batch_size)
            #    break

            if accumulated_steps == 0:
                if epoch_iter == num_iters:
                    break
                total_iter += 1
                epoch_iter += 1
                if iter_start_time is None:
                    iter_start_time = time.perf_counter()

                adjust_learning_rate(total_iter, optimizer, args.learning_rate,
                                     args.warmup_steps)

                model.zero_grad(set_to_none=True)

            x, y, num_frames = batch_to_gpu_me_char_with_wav(batch, use_tones=use_tones, use_wav=(args.train_wav and args.use_priorgrad))

            with torch.cuda.amp.autocast(enabled=args.amp):
                ###(mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred,
                # pitch_tgt, energy_pred, energy_tgt, attn_soft, attn_hard,
                #attn_hard_dur, attn_logprob, speaker_prediction, speaker, enc_out, self._reversal_classifier)
                y_pred = model(x, verbose=args.verbose)
                loss, meta = criterion(y_pred, y)

                if (args.kl_loss_start_epoch is not None
                        and epoch >= args.kl_loss_start_epoch):

                    if args.kl_loss_start_epoch == epoch and epoch_iter == 1:
                        print('Begin hard_attn loss')

                   #(mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred,
                   # pitch_tgt, energy_pred, energy_tgt, attn_soft, attn_hard,
                   # attn_hard_dur, attn_logprob, speaker_prediction, speaker, enc_out, self._reversal_classifier)
                    try:
                        _, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _ ,  _,_,_,_, _,_,_= y_pred
                    except:
                        _, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _ , _,_,_,_= y_pred


                    binarization_loss = attention_kl_loss(attn_hard, attn_soft)
                    kl_weight = min((epoch - args.kl_loss_start_epoch) / args.kl_loss_warmup_epochs, 1.0) * args.kl_loss_weight
                    meta['kl_loss'] = binarization_loss.clone().detach() * kl_weight
                    loss += kl_weight * binarization_loss

                else:
                    meta['kl_loss'] = torch.zeros_like(loss)
                    kl_weight = 0
                    binarization_loss = 0

                loss /= args.grad_accumulation

            meta = {k: v / args.grad_accumulation for k, v in meta.items()}

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if args.verbose and False:
                print("--->model: grad")
                for name, params in model.named_parameters():
                    print("-------->name:", name)
                    if params is not None and params.grad is not None:
                        print("-------------->max_grad:", params.grad.max(), "\t----->min_grad:", params.grad.min())
                    else:
                        print("----->isnone params ",params," params.grad ",params.grad)
                #print("--->decoder:")
                #for name, params in decoder.named_parameters():
                #    print("-->name:", name, "-->max_grad:", params.grad.max(), "-->min_grad:", params.grad.min())


            if distributed_run:
                reduced_loss = reduce_tensor(loss.data, args.world_size).item()
                reduced_num_frames = reduce_tensor(num_frames.data, 1).item()
                meta = {k: reduce_tensor(v, args.world_size) for k, v in meta.items()}
            else:
                reduced_loss = loss.item()
                reduced_num_frames = num_frames.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            accumulated_steps += 1
            iter_loss += reduced_loss
            iter_num_frames += reduced_num_frames
            iter_meta = {k: iter_meta.get(k, 0) + meta.get(k, 0) for k in meta}

            if accumulated_steps % args.grad_accumulation == 0:

                logger.log_grads_tb(total_iter, model)
                if args.amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_thresh)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_thresh)

                    #dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1)
                    optimizer.step()

                if args.ema_decay > 0.0:
                    apply_multi_tensor_ema(args.ema_decay, *mt_ema_params)

                iter_time = time.perf_counter() - iter_start_time
                iter_mel_loss = iter_meta['mel_loss'].item()
                iter_lang_loss = iter_meta['lang_loss'].item() if iter_meta['lang_loss'] !=0 else 0
                iter_kl_loss = iter_meta['kl_loss'].item()
                epoch_frames_per_sec[-1] += iter_num_frames / iter_time
                epoch_loss[-1] += iter_loss
                epoch_num_frames[-1] += iter_num_frames
                epoch_mel_loss[-1] += iter_mel_loss
                epoch_lang_loss[-1] += iter_lang_loss

                logger.log((epoch, epoch_iter, num_iters),
                           tb_total_steps=total_iter,
                           subset='train',
                           data=OrderedDict(
                             [
                               ('loss', iter_loss),
                               ('mel_loss', iter_mel_loss),
                               ('iter_lang_loss', iter_lang_loss),
                               ('kl_loss', iter_kl_loss),
                               ('kl_weight', kl_weight),
                               ('frames/s', iter_num_frames / iter_time),
                               ('took', iter_time),
                               ('lrate', optimizer.param_groups[0]['lr']),
                               #('decoder_grad_norm', dec_grad_norm)
                             ]),
                           )
                #logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                #                  global_step=iteration)
                accumulated_steps = 0
                iter_loss = 0
                iter_num_frames = 0
                iter_meta = {}
                iter_start_time = time.perf_counter()

        # Finished epoch
        epoch_loss[-1] /= epoch_iter
        epoch_mel_loss[-1] /= epoch_iter
        epoch_lang_loss[-1] /= epoch_iter
        epoch_time += [time.perf_counter() - epoch_start_time]
        iter_start_time = None

        logger.log((epoch,),
                   tb_total_steps=None,
                   subset='train_avg',
                   data=OrderedDict([
                       ('loss', epoch_loss[-1]),
                       ('mel_loss', epoch_mel_loss[-1]),
                       ('lang_loss', epoch_lang_loss[-1]),
                       ('frames/s', epoch_num_frames[-1] / epoch_time[-1]),
                       ('took', epoch_time[-1])]),
                   )

        validate_me(model, epoch, total_iter, criterion, valset, args.batch_size,
                 collate_fn, distributed_run, batch_to_gpu_me_char_with_wav,
                    use_tones=use_tones, use_wav=(args.train_wav and args.use_priorgrad),
                    obj_griffin_lim=mstft2, outwavdir=args.output, sampling_rate=args.sampling_rate)

        if args.ema_decay > 0:
            validate_me(ema_model, epoch, total_iter, criterion, valset,
                     args.batch_size, collate_fn, distributed_run, batch_to_gpu_me_char_with_wav,
                     ema=True, use_tones=use_tones, use_wav=(args.train_wav and args.use_priorgrad),
                        obj_griffin_lim=mstft2, outwavdir=args.output, sampling_rate=args.sampling_rate)

        maybe_save_checkpoint(args, model, ema_model, optimizer, scaler, epoch,
                              total_iter, model_config)
        logger.flush()

    # Finished training
    if args.pyprof:
        profiler.stop()
        torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)

    if len(epoch_loss) > 0:
        # Was trained - average the last 20 measurements
        last_ = lambda l: np.asarray(l[-20:])
        epoch_loss = last_(epoch_loss)
        epoch_mel_loss = last_(epoch_mel_loss)
        epoch_num_frames = last_(epoch_num_frames)
        epoch_time = last_(epoch_time)
        logger.log((),
                   tb_total_steps=None,
                   subset='train_avg',
                   data=OrderedDict([
                       ('loss', epoch_loss.mean()),
                       ('mel_loss', epoch_mel_loss.mean()),
                       ('frames/s', epoch_num_frames.sum() / epoch_time.sum()),
                       ('took', epoch_time.mean())]),
                   )

    validate_me(model, None, total_iter, criterion, valset, args.batch_size,
             collate_fn, distributed_run, batch_to_gpu_me_char_with_wav, use_tones=use_tones, use_wav=(args.train_wav and args.use_priorgrad))


if __name__ == '__main__':
    main()
