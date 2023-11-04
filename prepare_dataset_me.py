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
import time
from pathlib import Path

import numpy as np
import json
import torch
import tqdm
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from torch.utils.data import DataLoader

from fastpitch.data_function import TTSCollate, TTSDataset
from fastpitch.data_function_me import TTSDatasetMeMulti, TTSCollateMeMulti


#def parse_args(parser):
#    return parser


def main000():

    parser = argparse.ArgumentParser(description='FastPitch Data Pre-processing')
    #parser = parse_args(parser)
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--wav-text-filelists', required=True, nargs='+',
                        type=str, help='Files with audio paths and text')
    parser.add_argument('--extract-mels', action='store_true',
                        help='Calculate spectrograms from .wav files')
    parser.add_argument('--extract-pitch', action='store_true',
                        help='Extract pitch')
    parser.add_argument('--save-alignment-priors', action='store_true',
                        help='Pre-calculate diagonal matrices of alignment of text to audio')
    parser.add_argument('--log-file', type=str, default='preproc_log.json',
                         help='Filename for logging')
    parser.add_argument('--n-speakers', type=int, default=1)
    # Mel extraction
    parser.add_argument('--max-wav-value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--filter-length', default=1024, type=int,
                        help='Filter length')
    parser.add_argument('--hop-length', default=256, type=int,
                        help='Hop (stride) length')
    parser.add_argument('--win-length', default=1024, type=int,
                        help='Window length')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                        help='Minimum mel frequency')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                        help='Maximum mel frequency')
    parser.add_argument('--n-mel-channels', type=int, default=80)
    # Pitch extraction
    parser.add_argument('--f0-method', default='pyworld', type=str,
                        choices=['pyworld','pyin'], help='F0 estimation method')
    # Performance
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--n-workers', type=int, default=16)


    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, Path(args.dataset_path, args.log_file)),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k, v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k: v})
    DLLogger.flush()

    if args.extract_mels:
        Path(args.dataset_path, 'mels').mkdir(parents=False, exist_ok=True)

    if args.extract_pitch:
        Path(args.dataset_path, 'pitch').mkdir(parents=False, exist_ok=True)

    if args.save_alignment_priors:
        Path(args.dataset_path, 'alignment_priors').mkdir(parents=False, exist_ok=True)

    for filelist in args.wav_text_filelists:

        print(f'Processing {filelist}...')

        dataset = TTSDataset(
            args.dataset_path,
            filelist,
            text_cleaners=['english_cleaners_v2'],
            n_mel_channels=args.n_mel_channels,
            p_arpabet=0.0,
            n_speakers=args.n_speakers,
            load_mel_from_disk=False,
            load_pitch_from_disk=False,
            pitch_mean=None,
            pitch_std=None,
            max_wav_value=args.max_wav_value,
            sampling_rate=args.sampling_rate,
            filter_length=args.filter_length,
            hop_length=args.hop_length,
            win_length=args.win_length,
            mel_fmin=args.mel_fmin,
            mel_fmax=args.mel_fmax,
            betabinomial_online_dir=None,
            pitch_online_dir=None,
            pitch_online_method=args.f0_method)

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=args.n_workers,
            collate_fn=TTSCollate(),
            pin_memory=False,
            drop_last=False)

        all_filenames = set()
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            tik = time.time()

            _, input_lens, mels, mel_lens, _, pitch, _, _, attn_prior, fpaths = batch

            # Ensure filenames are unique
            for p in fpaths:
                fname = Path(p).name
                if fname in all_filenames:
                    raise ValueError(f'Filename is not unique: {fname}')
                all_filenames.add(fname)

            if args.extract_mels:
                for j, mel in enumerate(mels):
                    fname = Path(fpaths[j]).with_suffix('.pt').name
                    fpath = Path(args.dataset_path, 'mels', fname)
                    torch.save(mel[:, :mel_lens[j]], fpath)

            if args.extract_pitch:
                for j, p in enumerate(pitch):
                    fname = Path(fpaths[j]).with_suffix('.pt').name
                    fpath = Path(args.dataset_path, 'pitch', fname)
                    torch.save(p[:mel_lens[j]], fpath)

            if args.save_alignment_priors:
                for j, prior in enumerate(attn_prior):
                    fname = Path(fpaths[j]).with_suffix('.pt').name
                    fpath = Path(args.dataset_path, 'alignment_priors', fname)
                    torch.save(prior[:mel_lens[j], :input_lens[j]], fpath)


def main():

    parser = argparse.ArgumentParser(description='FastPitch Data Pre-processing')
    #parser = parse_args(parser)
    """
    Parse commandline arguments.
    """
    #parser.add_argument('-d', '--dataset-path', type=str,
    #                    default='./', help='Path to dataset')
    #parser.add_argument('--wav-text-filelists', required=True, nargs='+',
    #                    type=str, help='Files with audio paths and text')
    parser.add_argument(
        "--train-dir",
        default="dump/train",
        type=str,
        help="directory including training data. ",
    )
    #parser.add_argument(
    #    "--dev-dir",
    #    default="dump/valid",
    #    type=str,
    #    help="directory including development data. ",
    #)
    parser.add_argument(
        "--use-norm", default=1, type=int, help="usr norm-mels for train or raw."
    )
    parser.add_argument(
        "--f0-stat", default="./dump/stats_f0.npy", type=str, help="f0-stat path.",
    )
    parser.add_argument(
        "--energy-stat",
        default="./dump/stats_energy.npy",
        type=str,
        help="energy-stat path.",
    )
    parser.add_argument(
        "--dataset_config", default="preprocess/libritts_preprocess.yaml", type=str,
    )
    parser.add_argument(
        "--dataset_stats", default="dump/stats.npy", type=str,
    )
    parser.add_argument(
        "--dataset_mapping", default="dump/libritts_mapper.npy", type=str,
    )
    parser.add_argument(
        "--use_char", default=0, type=int,help="use char instead of phonemes"
    )
    parser.add_argument(
        "--tones", default=0, type=int,  help="use separeted tones"
    )
    parser.add_argument(
        "--toneseparated", default=0, type=int, help="use separeted tones"
    )

    parser.add_argument(
        "--convert_ipa", default=0, type=int,help="used shared ipa phoneme"
    )
    parser.add_argument(
        "--use_ipa_phone", default=0, type=int,help="used shared ipa phoneme"
    )
    parser.add_argument(
        "--load_language_array", default=0, type=int,help="use parameter generation network"
    )
    parser.add_argument(
        "--load_speaker_array", default=0, type=int,help="use parameter generation network"
    )
    
    parser.add_argument(
        "--filter_on_lang", default=0, type=int,help="can delete specific languages"
    )
    parser.add_argument(
        "--filter_on_speaker", default=0, type=int,help="can delete specific speakers"
    )
    parser.add_argument(
        "--filter_on_uttid", default=0, type=int, help="can delete specific utt_ids"
    )
    parser.add_argument(
        "--filter_remove_lang_ids", default=None, type=str, help="comma separated list of ids of language to delete"
    )
    parser.add_argument(
        "--filter_remove_speaker_ids", default=None, type=str,help="comma separated list of ids of speaker to delete"
    )
    parser.add_argument(
        "--filter_remove_uttid_ids", default=None, type=str,help="comma separated list of ids of utt to delete"
    )
    parser.add_argument(
        "--format", default="npy", type=str,help="format des fichiers"
    )

    parser.add_argument('--extract-mels', action='store_true', help='Calculate spectrograms from .wav files')
    parser.add_argument('--extract-pitch', action='store_true',  help='Extract pitch')
    parser.add_argument('--save-alignment-priors', action='store_true',   help='Pre-calculate diagonal matrices of alignment of text to audio')
    parser.add_argument('--log-file', type=str, default='preproc_log.json',   help='Filename for logging')

    #parser.add_argument('--n-speakers', type=int, default=1)
    # Mel extraction
    parser.add_argument('--max-wav-value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--sampling-rate', default=16000, type=int,
                        help='Sampling rate')
    parser.add_argument('--filter-length', default=1024, type=int,
                        help='Filter length')
    parser.add_argument('--hop-length', default=256, type=int,
                        help='Hop (stride) length')
    parser.add_argument('--win-length', default=1024, type=int,
                        help='Window length')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                        help='Minimum mel frequency')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                        help='Maximum mel frequency')
    parser.add_argument('--n-mel-channels', type=int, default=80)
    # Pitch extraction
    parser.add_argument('--f0-method', default='pyin', type=str,
                        choices=['pyin'], help='F0 estimation method')
    # Performance
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--n-workers', type=int, default=16)


    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, Path(args.train_dir, args.log_file)),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k, v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k: v})
    DLLogger.flush()


    args.use_norm = bool(args.use_norm)
    args.use_char=bool(args.use_char)
    args.convert_ipa=bool(args.convert_ipa)
    args.use_ipa_phone= bool(args.use_ipa_phone)
    args.load_speaker_array= bool(args.load_speaker_array)
    args.load_language_array= bool(args.load_language_array)
    args.filter_on_lang=bool(args.filter_on_lang)
    args.filter_on_speaker=bool(args.filter_on_speaker)
    args.filter_on_uttid=bool(args.filter_on_uttid)
    args.tones= bool(args.tones)
    args.toneseparated= bool(args.toneseparated)

    use_tones=False
    if args.tones or args.toneseparated:
        use_tones=True
    #print("### args args.filter_remove_lang_ids ", args.filter_remove_lang_ids)
    if args.filter_remove_lang_ids is not None:
        args.filter_remove_lang_ids=[int(i) if str(i).isnumeric() else i   for i in args.filter_remove_lang_ids.split(',')] 
    if args.filter_remove_speaker_ids is not None:
        args.filter_remove_speaker_ids=[int(i) if str(i).isnumeric() else i for i in args.filter_remove_speaker_ids.split(',')] 
    if args.filter_remove_uttid_ids is not None:
        args.filter_remove_uttid_ids=[i for i in args.filter_remove_uttid_ids.split(',')]

    if args.format == "npy":
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
    elif args.format=="pt":
        wave_query= "*-wave.pt"
        charactor_query = "*-ids.pt"
        tones_query = "*-tonesids.pt"

        speaker_query = "*-speaker-ids.pt"
        language_query = "*-language-ids.pt"
        language_speaker_query = "*-language-speaker-ids.pt"

        language_array_query="*-languagearray-ids.pt"
        speaker_array_query="*-speakerarray-ids.pt"

        mel_query = "*-raw-feats.pt" if args.use_norm is False else "*-norm-feats.pt"

        duration_query = "*-durations.pt"
        f0_query = "*-raw-f0.pt"
        energy_query = "*-raw-energy.pt"
        speaker_load_fn = torch.load
        language_load_fn = torch.load
    else:
        raise ValueError("Only npy are supported.")

    speakers_map = None
    languages_map = None

    with open(args.dataset_mapping) as f:
        dataset_mapping = json.load(f)
        speakers_map = dataset_mapping["speakers_map"]
        languages_map = dataset_mapping["languages_map"]

    n_speakers=len(speakers_map)
    n_languages= len(languages_map)

    convert_ipa=False
    if args.convert_ipa:
        convert_ipa=True
    if args.use_ipa_phone:
        convert_ipa=False

    if args.extract_mels:
        Path(args.train_dir, 'mels').mkdir(parents=False, exist_ok=True)

    if args.extract_pitch:
        Path(args.train_dir, 'pitch').mkdir(parents=False, exist_ok=True)

    if args.save_alignment_priors:
        Path(args.train_dir, 'alignment_priors').mkdir(parents=False, exist_ok=True)

    mel_length_threshold= 48
    max_mel_length_threshold=100*mel_length_threshold

    if True:
    #for filelist in args.wav_text_filelists:

        #print(f'Processing {filelist}...')

        dataset = TTSDatasetMeMulti(
            #args.dataset_path,
            #filelist,
            root_dir=args.train_dir,
            wave_query=wave_query, charactor_query=charactor_query, tones_query=tones_query,
            speaker_query=speaker_query,
            language_speaker_query=language_speaker_query,
            language_query=language_query,
            charactor_load_fn=np.load, language_load_fn=language_load_fn,
            speaker_load_fn=speaker_load_fn,
            mel_query=mel_query, mel_load_fn=np.load,
            duration_query=duration_query, duration_load_fn=np.load,
            f0_query=f0_query, f0_load_fn=np.load, pitch_query=pitch_query,
            energy_query=energy_query, energy_load_fn=np.load,
            f0_stat=args.f0_stat,
            energy_stat=args.energy_stat,
            mel_length_threshold=mel_length_threshold, max_mel_length_threshold=max_mel_length_threshold,
            speakers_map=speakers_map,
            languages_map=languages_map, convert_ipa=convert_ipa, use_tones=use_tones,
            filter_on_lang=args.filter_on_lang,
            filter_on_speaker=args.filter_on_speaker,
            filter_on_uttid=args.filter_on_uttid,
            filter_remove_lang_ids=args.filter_remove_lang_ids,        
            filter_remove_speaker_ids=args.filter_remove_speaker_ids,   
            filter_remove_uttid_ids=args.filter_remove_uttid_ids, 
            load_language_array=args.load_language_array , load_speaker_array=args.load_speaker_array,
            language_array_query=language_array_query,   speaker_array_query=speaker_array_query,

                # config=None,
                 n_speakers=n_speakers, n_languages=n_languages,
                
                #text_cleaners=['english_cleaners_v2'],
                n_mel_channels=args.n_mel_channels,
                p_arpabet=0.0,
                ####n_speakers=args.n_speakers,
                load_mel_from_disk=False, load_my_mel_from_disk=False, load_my_duration_from_disk= False,
                load_pitch_from_disk=False, load_energy_from_disk=False, load_f0_from_disk=False,
                pitch_mean=None,
                pitch_std=None,
                max_wav_value=args.max_wav_value,
                sampling_rate=args.sampling_rate,
                filter_length=args.filter_length,
                hop_length=args.hop_length,
                win_length=args.win_length,
                mel_fmin=args.mel_fmin,
                mel_fmax=args.mel_fmax,
                betabinomial_online_dir=None,
                pitch_online_dir=None,
                pitch_online_method=args.f0_method)

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=args.n_workers,
            collate_fn=TTSCollateMeMulti(),
            pin_memory=False,
            drop_last=False)

        all_filenames = set()
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            tik = time.time()

            #_, input_lens, mels, mel_lens, _, pitch, _, _, attn_prior, fpaths = batch
            if use_tones:
               #(text_padded, tones_padded, input_lengths, mel_padded, output_lengths, len_x, pitch_padded, energy_padded, speaker, language,  attn_prior_padded, utt_id)
                _, _, input_lens, mels, mel_lens, _, pitch, _, _, _, attn_prior, utt_ids = batch
            else:
                _, input_lens, mels, mel_lens, _, pitch, _, _, _, attn_prior, utt_ids = batch
            # Ensure filenames are unique
            for p in utt_ids:
                fname = p #Path(p).name
                if fname in all_filenames:
                    raise ValueError(f'Filename is not unique: {fname}')
                all_filenames.add(fname)

            if args.extract_mels:
                for j, mel in enumerate(mels):
                    fname = Path(utt_ids[j]).with_suffix('.pt').name
                    fpath = Path(args.train_dir, 'mels', fname)
                    torch.save(mel[:, :mel_lens[j]], fpath)

            if args.extract_pitch:
                for j, p in enumerate(pitch):
                    fname = Path(utt_ids[j]).with_suffix('.pt').name
                    fpath = Path(args.train_dir, 'pitch', fname)
                    torch.save(p[:mel_lens[j]], fpath)

            if args.save_alignment_priors:
                for j, prior in enumerate(attn_prior):
                    fname = Path(utt_ids[j]).with_suffix('.pt').name
                    fpath = Path(args.train_dir, 'alignment_priors', fname)
                    torch.save(prior[:mel_lens[j], :input_lens[j]], fpath)


if __name__ == '__main__':
    main()
