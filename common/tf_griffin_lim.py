# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Griffin-Lim phase reconstruction algorithm from mel spectrogram."""

import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def griffin_lim_lb(
    mel_spec, stats_path,  n_iter=32, output_dir=None, wav_name="lb", sampling_rate=16000, fft_size=1024,  num_mels=80, 
fmin=0, fmax=8000,  hop_size=256, win_length=1024):
    """Generate wave from mel spectrogram with Griffin-Lim algorithm using Librosa.
    Args:
        mel_spec (ndarray): array representing the mel spectrogram.
        stats_path (str): path to the `stats.npy` file containing norm statistics.
        dataset_config (Dict): dataset configuration parameters.
        n_iter (int): number of iterations for GL.
        output_dir (str): output directory where audio file will be saved.
        wav_name (str): name of the output file.
    Returns:
        gl_lb (ndarray): generated wave.
    """
    try:
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = np.load(stats_path)
    
        mel_spec = np.power(10.0, scaler.inverse_transform(mel_spec)).T
        mel_basis = librosa.filters.mel(
            hop_size,
            n_fft=fft_size,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        mel_to_linear = np.maximum(1e-10, np.dot(np.linalg.pinv(mel_basis), mel_spec))
        gl_lb = librosa.griffinlim(
            mel_to_linear,
            n_iter=n_iter,
            hop_length=hop_size,
            win_length=win_length,# or fft_size,
        )
        if output_dir:
            output_path = os.path.join(output_dir, f"{wav_name}.wav")
            sf.write(output_path, gl_lb, hop_size, "PCM_16")
        return gl_lb
    except ValueError as error:
        print(error)
        return None



class TFGriffinLim(tf.keras.layers.Layer):
    """Griffin-Lim algorithm for phase reconstruction from mel spectrogram magnitude."""

    def __init__(self, stats_path, normalized = True, sampling_rate=16000, fft_size=1024, num_mels=80, fmin=0, fmax=8000, win_length=1024, hop_size=256):
        """Init GL params.
        Args:
            stats_path (str): path to the `stats.npy` file containing norm statistics.
            dataset_config (Dict): dataset configuration parameters.
        """
        super().__init__()
        self.sampling_rate=sampling_rate,
        self.fft_size=fft_size, 
        self.num_mels=num_mels, 
        self.fmin=fmin 
        self.fmax=fmax 
        self.win_length=win_length 
        self.hop_size=hop_size

        self.normalized = normalized
        if normalized:
            scaler = StandardScaler()
            scaler.mean_, scaler.scale_ = np.load(stats_path)
            self.scaler = scaler
        #self.ds_config = dataset_config
        self.mel_basis = librosa.filters.mel(
            sampling_rate,
            n_fft=fft_size,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )  # [num_mels, fft_size // 2 + 1]

    def save_wav(self, gl_tf, output_dir, wav_name):
        """Generate WAV file and save it.
        Args:
            gl_tf (tf.Tensor): reconstructed signal from GL algorithm.
            output_dir (str): output directory where audio file will be saved.
            wav_name (str): name of the output file.
        """
        encode_fn = lambda x: tf.audio.encode_wav(x, self.sampling_rate)
        gl_tf = tf.expand_dims(gl_tf, -1)
        if not isinstance(wav_name, list):
            wav_name = [wav_name]

        if len(gl_tf.shape) > 2:
            bs, *_ = gl_tf.shape
            assert bs == len(wav_name), "Batch and 'wav_name' have different size."
            tf_wav = tf.map_fn(encode_fn, gl_tf, dtype=tf.string)
            for idx in tf.range(bs):
                output_path = os.path.join(output_dir, f"{wav_name[idx]}.wav")
                tf.io.write_file(output_path, tf_wav[idx])
        else:
            tf_wav = encode_fn(gl_tf)
            tf.io.write_file(os.path.join(output_dir, f"{wav_name}.wav"), tf_wav)


    def call(self, mel_spec, n_iter=32):
        """Apply GL algorithm to batched mel spectrograms.
        Args:
            mel_spec (tf.Tensor): normalized mel spectrogram.
            n_iter (int): number of iterations to run GL algorithm.
        Returns:
            (tf.Tensor): reconstructed signal from GL algorithm.
        """
        # de-normalize mel spectogram
        if self.normalized:
            mel_spec = tf.math.pow(
                10.0, mel_spec * self.scaler.scale_ + self.scaler.mean_
            )
        else:
            mel_spec = tf.math.pow(
                10.0, mel_spec
            )  # TODO @dathudeptrai check if its ok without it wavs were too quiet
        inverse_mel = tf.linalg.pinv(self.mel_basis)

        # [:, num_mels] @ [fft_size // 2 + 1, num_mels].T
        mel_to_linear = tf.linalg.matmul(mel_spec, inverse_mel, transpose_b=True)
        mel_to_linear = tf.cast(tf.math.maximum(1e-10, mel_to_linear), tf.complex64)

        init_phase = tf.cast(
            tf.random.uniform(tf.shape(mel_to_linear), maxval=1), tf.complex64
        )
        
        phase = tf.math.exp(2j * np.pi * init_phase)

        _windows_fn = tf.signal.inverse_stft_window_fn(self.hop_size)
        # tf.signal.inverse_stft(stfts, frame_length, frame_step, fft_length=None, window_fn=tf.signal.hann_window, name=None)
        for _ in tf.range(n_iter):
            inverse = tf.signal.inverse_stft(
                mel_to_linear * phase,
                frame_length=self.win_length or self.fft_size,
                frame_step=self.hop_size,
                fft_length=self.fft_size,
                window_fn=_windows_fn,
            )
            phase = tf.signal.stft(
                inverse,
                self.win_length or self.fft_size,
                self.hop_size,
               None, # self.fft_size,
            )
            phase /= tf.cast(tf.maximum(1e-10, tf.abs(phase)), tf.complex64)

        return tf.signal.inverse_stft(
            mel_to_linear * phase,
            frame_length=self.win_length or self.fft_size,
            frame_step=self.hop_size,
            fft_length= self.fft_size,
            window_fn=_windows_fn, #tf.signal.inverse_stft_window_fn(self.hop_size),
        )



def save_wav_audio_from_spectogram(mel_gt, outdir, utt_id,  dataset_stats, sampling_rate=16000, fft_size=1024, num_mels=80, fmin=0, fmax=8000, win_length=1024, hop_size=256,  randsav=0):

    #mel_gt= np.squeeze(mel_gt).transpose(1, 0)
    griffin_lim_tf = TFGriffinLim(stats_path=dataset_stats, sampling_rate=sampling_rate, fft_size=fft_size, num_mels=num_mels, fmin=fmin, fmax=fmax, win_length=win_length, hop_size=hop_size, normalized=False)

    griff_dir_name = os.path.join( outdir, "griffin" )

    #Path(outdir, 'griffin').mkdir(parents=False, exist_ok=True)
    if not os.path.exists(griff_dir_name):
        os.makedirs(griff_dir_name)

    if True:
        mel_mod=np.squeeze(mel_gt)
        mel_mod_tf= tf.reshape(tf.convert_to_tensor(mel_mod), [-1, 80])[tf.newaxis, :]

        grif_gt2 = griffin_lim_lb( tf.convert_to_tensor(mel_mod), stats_path=dataset_stats,  n_iter=32, output_dir=griff_dir_name, wav_name=f"{utt_id}_{randsav}_gt2", sampling_rate=sampling_rate, fft_size=fft_size,  num_mels=num_mels, fmin=fmin, fmax=fmax,  hop_size=hop_size, win_length=win_length)

        grif_gt = griffin_lim_tf(mel_mod_tf, n_iter=32 )
        #grif_gt = griffin_lim_tf(tf.reshape(np.squeeze(mel_gt), [-1, 80])[tf.newaxis, :], n_iter=32 )
        print(" griffin grif_gt ", tf.shape(grif_gt))
        griffin_lim_tf.save_wav(grif_gt, griff_dir_name, f"{utt_id}_{randsav}_gt")






