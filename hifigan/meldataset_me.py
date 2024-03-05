import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from meldataset import *

from stft import STFT, TacotronSTFT


#MAX_WAV_VALUE = 32768.0
#mel_basis = {}
#hann_window = {}





class MelDataset_me(torch.utils.data.Dataset):
    def __init__(self, 
        root_dir,
        audio_query="*-wave.npy",
        audio2_query="*.wav",
        mel_query="*-raw-feats.npy",
        audio_dir="wavs", audio2_dir="", mel_dir="",
        audio_load_fn=np.load, 
        mel_load_fn=np.load,
        audio_length_threshold=0,
        mel_length_threshold=0,  load_my_mel_file= False,
        filter_length=1024, 

       #training_files, 
         segment_size=160000, n_fft=1024, num_mels=80,
                 hop_size=256, win_size=1024, sampling_rate=16000,  fmin=0.0, fmax=8000.0, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False):  #, base_mels_path=None
        #self.audio_files = training_files

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        #self.sampling_rate = sampling_rate



        self.audio_files = sorted(find_files(os.path.join(root_dir, audio_dir), audio_query)+find_files(os.path.join(root_dir, audio2_dir), audio2_query))
        self.mel_files = sorted(find_files(os.path.join(root_dir, mel_dir), mel_query))
        # assert the number of files
        assert len(self.audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(self.audio_files) == len(
            self.mel_files
        ), f"Number of audio and mel files are different ({len(audio_files)} vs {len(mel_files)})."

        if  ".wav" in audio_query or ".npy" in audio_query or ".pt" in audio_query or ".pt" in audio2_query or ".npy" in audio2_query or ".wav" in audio2_query:
            suffix1 = audio_query[1:]
            suffix2 = audio2_query[1:]
            suffix_mel = mel_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix1, "").replace(suffix2, "") for f in self.audio_files]
            utt_ids_mel = [os.path.basename(f).replace(suffix_mel, "") for f in self.mel_files]

        #print("utt_ids ", utt_ids[0:10])
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)

        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_size, n_mel_channels=num_mels,
                                 win_length=win_size,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=fmin, mel_fmax=fmax)


        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        #self.base_mels_path = base_mels_path
        self.load_my_mel_file=load_my_mel_file
        self.utt_ids = utt_ids
        self.utt_ids_mel = utt_ids_mel
        #self.audio_files = audio_files
        #self.mel_files = mel_files
        self.audio_load_fn = audio_load_fn
        self.mel_load_fn = mel_load_fn
        self.audio_length_threshold = audio_length_threshold
        self.mel_length_threshold = mel_length_threshold
        self.segment_length = (segment_size//self.hop_size)*self.hop_size

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec


    def __getitem__(self, index):
        filename = self.audio_files[index]
        utt_id=self.utt_ids[index]
        #filename = self.audio_files[index]
        if len(self.audio_files)==len(  self.mel_files   ):
            melfilename= self.mel_files[index] if len(self.mel_files)>index else None
            #print("######SAME SIZE  ",utt_id,  melfilename)
        else:
            utt_mel_id=[i for i,n  in enumerate(self.utt_ids_mel)  if utt_id==n]
            melfilename= self.mel_files[utt_mel_id[0]] if len(utt_mel_id)>0 else None
            #print("######SAME SIZE  ",utt_id,  melfilename)




        if self._cache_ref_count == 0:
            
            if os.path.splitext(filename)[1].strip().lower()=='.wav':
                #print("######here0 ", filename)
                audio, sampling_rate = load_wav(filename) #audio, sampling_rate = load_wav_to_torch(filename)
            else:
                #audio, sampling_rate = np.load(filename), self.sampling_rate
                #print("######here1 ", filename)
                audio= np.load(filename)
                #audio= torch.from_numpy(np.asarray(audio)).float()
                sampling_rate = self.sampling_rate


            audio = audio / MAX_WAV_VALUE

            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))

            #if sampling_rate != self.sampling_rate:
            #    raise ValueError("{} SR doesn't match target {} SR".format(
            #        sampling_rate, self.sampling_rate))


            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning and not  self.load_my_mel_file:
            #print(" not self.fine_tuning and not  self.load_my_mel_file ")
            if self.split:
                #print("splitting")
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    #print(" no splitting")
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False) #mel = self.get_mel(audio)
            #print("##not fine_tuning audio=",audio.size(),"  mel=",mel.size())#,"",, "",,,)

        else:
            if os.path.basename(melfilename).endswith(".npy"):
                mel= np.load(melfilename)
                mel= torch.from_numpy(np.asarray(mel)).float()                
                #mel = np.load( os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
                #mel = torch.from_numpy(mel)
            else: 
                mel = torch.load(melfilename)
                mel = torch.FloatTensor(mel)


            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            #print("##fine_tuning audio=",audio.size(),"  mel=",mel.size())#,"",, "",,,)


        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        #print(" audio=",audio.size(),"  mel=",mel.size())#,"",, "",,,)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
