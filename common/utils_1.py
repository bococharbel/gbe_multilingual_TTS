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

import shutil
import warnings
from pathlib import Path
from typing import Optional

import fnmatch
import os
import librosa
import numpy as np

import torch
from scipy.io.wavfile import read

###
import numpy as np
import librosa
import os, copy
from scipy import signal
from hyperparams import hparams as hp
import soundfile


def average_by_duration(x, durs):
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))

    # calculate charactor f0/energy
    x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0  # np.mean([]) = nan.

    return x_char.astype(np.float32)


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.
    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.
    Returns:
        list: List of found filenames.
    """
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files



def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


def load_wav_to_torch(full_path, force_sampling_rate=None):
    if os.path.basename(full_path).endswith(".npy"):
        data = np.load(full_path)
        sampling_rate= force_sampling_rate
    elif force_sampling_rate is not None:
        data, sampling_rate = librosa.load(full_path, sr=force_sampling_rate)
    else:
        sampling_rate, data = read(full_path)

    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(dataset_path, fnames, has_speakers=False, split="|"):
    def split_line(root, line):
        parts = line.strip().split(split)
        if has_speakers:
            paths, non_paths = parts[:-2], parts[-2:]
        else:
            paths, non_paths = parts[:-1], parts[-1:]
        return tuple(str(Path(root, p)) for p in paths) + tuple(non_paths)

    fpaths_and_text = []
    for fname in fnames:
        with open(fname, encoding='utf-8') as f:
            fpaths_and_text += [split_line(dataset_path, line) for line in f]
    return fpaths_and_text


def stats_filename(dataset_path, filelist_path, feature_name):
    stem = Path(filelist_path).stem
    return Path(dataset_path, f'{feature_name}_stats__{stem}.json')


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def to_device_async(tensor, device):
    return tensor.to(device, non_blocking=True)


def to_numpy(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


def prepare_tmp(path):
    if path is None:
        return
    p = Path(path)
    if p.is_dir():
        warnings.warn(f'{p} exists. Removing...')
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=False, exist_ok=False)

#https://github.com/mutiann/few-shot-transformer-tts/blob/main/utils/audio.py

def get_mel_basis():
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.num_mels)  # (n_mels, 1+n_fft//2)
    return _mel_basis

#https://github.com/mutiann/few-shot-transformer-tts/blob/main/utils/audio.py

def get_spectrograms(wav):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      wav: A 1d array of normalized and trimmed waveform.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
    '''
    y = wav

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = get_mel_basis()
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    if hp.symmetric_mel:
        mel = mel * hp.max_abs_value * 2 - hp.max_abs_value

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)

    return mel

#https://github.com/mutiann/few-shot-transformer-tts/blob/main/utils/audio.py

def mel_to_linear(mel):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(get_mel_basis())
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel))

#https://github.com/mutiann/few-shot-transformer-tts/blob/main/utils/audio.py

def mel2wav(mel):
    if hp.symmetric_mel:
        mel = (mel.T + hp.max_abs_value) / (2 * hp.max_abs_value)
    # de-noramlize
    mel = (np.clip(mel, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mel = np.power(10.0, mel * 0.05)
    mel = mel_to_linear(mel)

    # wav reconstruction
    wav = griffin_lim(mel**hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    return wav.astype(np.float32)

#https://github.com/mutiann/few-shot-transformer-tts/blob/main/utils/audio.py

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

#https://github.com/mutiann/few-shot-transformer-tts/blob/main/utils/audio.py

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

#https://github.com/mutiann/few-shot-transformer-tts/blob/main/utils/audio.py

def load_wav(path):
    return librosa.core.load(path, sr=hp.sr)[0]


#https://github.com/mutiann/few-shot-transformer-tts/blob/main/utils/audio.py
def save_wav(wav, path):
    wav_ = wav * 1 / max(0.01, np.max(np.abs(wav)))
    soundfile.write(path, wav_.astype(np.float32), hp.sr)
    return path

#https://github.com/mutiann/few-shot-transformer-tts/blob/main/utils/audio.py

def trim_silence_intervals(wav):
    intervals = librosa.effects.split(wav, top_db=50,
                                    frame_length=int(hp.sr / 1000 * hp.frame_length_ms) * 8,
                                    hop_length=int(hp.sr / 1000 * hp.frame_shift_ms))
    wav = np.concatenate([wav[l: r] for l, r in intervals])
    return wav



###########https://github.com/jik876/hifi-gan################"

import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
