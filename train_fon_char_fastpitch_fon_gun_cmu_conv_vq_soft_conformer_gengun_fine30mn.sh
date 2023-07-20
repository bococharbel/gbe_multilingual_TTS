#!/usr/bin/env bash

export OMP_NUM_THREADS=1

: ${NUM_GPUS:=1}
: ${BATCH_SIZE:=4}
: ${GRAD_ACCUMULATION:=2}
#: ${OUTPUT_DIR:="/scratch/roseline/dump/DumpFongbefrall/fastpitchoutput"}
: ${OUTPUT_DIR:="/media/phifamin/DATA/mebenit/test2/train_fastspeech_conv_vq_soft3_conformer_fine30mn"}
: ${LOG_FILE:=$OUTPUT_DIR/nvlog.json}
#: ${DATASET_PATH:=LJSpeech-1.1}
#: ${TRAIN_FILELIST:=filelists/ljs_audio_pitch_text_train_v3.txt}
#: ${VAL_FILELIST:=filelists/ljs_audio_pitch_text_val.txt}
: ${AMP:=false}
: ${SEED:=""}
: ${CHECKPOINT_PATH:="/media/phifamin/DATA/mebenit/test2/train_fastspeech_conv_vq_soft3_conformer/FastPitchMultiwavVq_checkpoint_230.pt"}
#: ${VARIANT:=TACOTRON2}
: ${LEARNING_RATE:=0.01}
: ${$PITCH_ONLINE_METHOD:=0.01}

# Adjust these when the amount of data changes
: ${EPOCHS:=350}
: ${EPOCHS_PER_CHECKPOINT:=50}
: ${WARMUP_STEPS:=1}
: ${KL_LOSS_WARMUP:=10000}

# Train a mixed phoneme/grapheme model
: ${PHONE:=true}
# Enable energy conditioning
: ${ENERGY:=true}
: ${TEXT_CLEANERS:=english_cleaners_v2}
# Add dummy space prefix/suffix is audio is not precisely trimmed
: ${APPEND_SPACES:=false}

#: ${LOAD_PITCH_FROM_DISK:=false}
#: ${LOAD_MEL_FROM_DISK:=false}

# For multispeaker models, add speaker ID = {0, 1, ...} as the last filelist column
#: ${NSPEAKERS:=33}
: ${SAMPLING_RATE:=16000}

#: ${TRAIN_DIR:=/scratch/roseline/dump/DumpFongbefrall/train/}
#: ${DEV_DIR:=/scratch/roseline/dump/DumpFongbefrall/valid/}
#: ${F0_STATS:=/scratch/roseline/dump/DumpFongbefrall/stats_f0.npy}
#: ${ENERGY_STATS:=/scratch/roseline/dump/DumpFongbefrall/stats_energy.npy}
#: ${DATASET_MAPPING:=/scratch/roseline/dump/DumpFongbefrall/fr_fon_mapper.json}
##: ${DATASET_CONFIG:=/scratch/roseline/dump/TensorFlowTTS/preprocess/preprocess_francaisfongbesmultispeakers.yaml }
#: ${DATASET_STATS:=/scratch/roseline/dump/DumpFongbefrall/stats.npy}
: ${TRAIN_DIR:=/media/phifamin/DATA/mebenit/test2/}
: ${DEV_DIR:=/media/phifamin/DATA/mebenit/test2/}
: ${F0_STATS:=/home/roseline/benit/DumpFongbefrall/stats_f0.npy}
: ${ENERGY_STATS:=/home/roseline/benit/DumpFongbefrall/stats_energy.npy}
#: ${DATASET_MAPPING:=/home/roseline/benit/DumpFongbefrall/fr_fon_mapper.json}
: ${DATASET_MAPPING:=/media/phifamin/DATA/mebenit/test2/mapper_fon_yor_gun_gen_char.json}
#: ${DATASET_CONFIG:=/home/roseline/benit/TensorFlowTTS/preprocess/preprocess_francaisfongbesmultispeakers.yaml }
: ${DATASET_STATS:=/home/roseline/benit/DumpFongbefrall/stats.npy}

#: ${TRAIN_META:=/media/phifamin/DATA/mebenit/test2/metadata_fon_gun_yor_mina_cmu_train.csv}
: ${TRAIN_META:=/media/phifamin/DATA/mebenit/test2/metadata_gengbe_gungbe_30mn_x2_cmu_train.csv}
#metadata_gengbe_gungbe_2h_8k_2_cmu_train.csv
 #metadata_gengbe_gungbe_30mn_2k_cmu_train.csv
 #metadata_gengbe_gungbe_1h_2k_cmu_train.csv
 #metadata_gengbe_gungbe_1h_4k_cmu_train.csv
 #metadata_gengbe_gungbe_1h_6k_cmu_train.csv
 #metadata_gengbe_gungbe_2h_4k_2_cmu_train.csv
 #metadata_gengbe_gungbe_2h_6k_2_cmu_train.csv
#metadata_gengbe_gungbe_1h_x2_cmu_train.csv
#metadata_gengbe_gungbe_2h_x2_cmu_train.csv
#metadata_gungbe_1h_cmu_train.csv
#metadata_gengbe_gungbe_15mn_x2_cmu_train.csv
#metadata_gengbe_gungbe_30mn_x2_cmu_train.csv
#metadata_gungbe_2h_cmu_train.csv
#metadata_gungbe_30mn_cmu_train.csv
#metadata_gungbe_15mn_cmu_train.csv
#metadata_gungbe_3h_cmu_train.csv
#metadata_gungbe_matie_luc_jean_cmu_train.csv
#metadata_gengbe_15mn_cmu_train.csv
#metadata_gengbe_2h_cmu_train.csv
#metadata_gengbe_1h_cmu_train.csv
#metadata_gengbe_30mn_cmu_train.csv
#metadata_gengbe_3h_cmu_train.csv

: ${VALID_META:=/media/phifamin/DATA/mebenit/test2/metadata_fon_gun_yor_mina_cmu_valid.csv}

: ${USE_CHAR:=0}
: ${TONES:=0}
: ${TONESSEPARATED:=0}
#: ${USE_IPA:=0}
: ${USE_NORM:=1}
: ${CONVERT_IPA:=0}
: ${USE_IPA_PHONE:=0}
: ${LOAD_LANGUAGE_ARRAY:=0}
: ${LOAD_SPEAKER_ARRAY:=0}

: ${FILTER_ON_LANG:=0}
: ${FILTER_ON_SPEAKER:=0}
: ${FILTER_ON_UTTID:=0}
: ${REMOVE_LANG_IDS:=gengbe,gungbe,adjagbe}
#: ${REMOVE_LANG_IDS:=english,french,yoruba}

: ${MTYPE:=2}

: ${GENERATED:=0}
#: ${TRAIN_TYPE:=fastpitch}
: ${TRAIN_TYPE:=create}
: ${N_MEL_CHANNELS:=80}
: ${FORMAT:=pt}
: ${N_SYMBOLS:=300}
: ${PADDING_IDX:=0}
: ${SYMBOLS_EMBEDDING_DIM:=384}


# Adjust env variables to maintain the global batch size: NUM_GPUS x BATCH_SIZE x GRAD_ACCUMULATION = 256.
GBS=$(($NUM_GPUS * $BATCH_SIZE * $GRAD_ACCUMULATION))
[ $GBS -ne 256 ] && echo -e "\nWARNING: Global batch size changed from 256 to ${GBS}."
echo -e "\nAMP=$AMP, ${NUM_GPUS}x${BATCH_SIZE}x${GRAD_ACCUMULATION}" \
        "(global batch size ${GBS})\n"

ARGS=""
ARGS+=" --cuda "
#ARGS+="  --reversal-classifier"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
#ARGS+=" --dataset-path $DATASET_PATH"
#ARGS+=" --training-files $TRAIN_FILELIST"
#ARGS+=" --validation-files $VAL_FILELIST"
ARGS+=" -bs $BATCH_SIZE"
ARGS+=" --grad-accumulation $GRAD_ACCUMULATION"
ARGS+=" --optimizer lamb"
ARGS+=" --epochs $EPOCHS"
ARGS+=" --epochs-per-checkpoint $EPOCHS_PER_CHECKPOINT"
#ARGS+=" --resume"
ARGS+=" --warmup-steps $WARMUP_STEPS"
ARGS+=" -lr $LEARNING_RATE"
ARGS+=" --weight-decay 1e-8"
ARGS+=" --grad-clip-thresh 1000.0"
ARGS+=" --dur-predictor-loss-scale 0.1"
ARGS+=" --pitch-predictor-loss-scale 0.1"

# Autoalign & new features
ARGS+=" --kl-loss-start-epoch 0"
ARGS+=" --kl-loss-warmup-epochs $KL_LOSS_WARMUP"
ARGS+=" --text-cleaners $TEXT_CLEANERS"
#ARGS+=" --n-speakers $NSPEAKERS"
#ARGS+=" --n-speakers $NSPEAKERS"
ARGS+=" --nmel-channels $N_MEL_CHANNELS"

[ "$AMP" = "true" ]                && ARGS+=" --amp"
[ "$PHONE" = "true" ]              && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]             && ARGS+=" --energy-conditioning"
[ "$SEED" != "" ]                  && ARGS+=" --seed $SEED"
#[ "$LOAD_MEL_FROM_DISK" = true ]   && ARGS+=" --load-mel-from-disk"
#[ "$LOAD_PITCH_FROM_DISK" = true ] && ARGS+=" --load-pitch-from-disk"
[ "$PITCH_ONLINE_DIR" != "" ]      && ARGS+=" --pitch-online-dir $PITCH_ONLINE_DIR"  # e.g., /dev/shm/pitch
[ "$PITCH_ONLINE_METHOD" != "" ]   && ARGS+=" --pitch-online-method $PITCH_ONLINE_METHOD"
#[ "$APPEND_SPACES" = true ]        && ARGS+=" --prepend-space-to-text"
#[ "$APPEND_SPACES" = true ]        && ARGS+=" --append-space-to-text"

#if [ "$SAMPLING_RATE" == "44100" ]; then
#  ARGS+=" --sampling-rate 44100"
#  ARGS+=" --filter-length 2048"
#  ARGS+=" --hop-length 512"
#  ARGS+=" --win-length 2048"
#  ARGS+=" --mel-fmin 0.0"
#  ARGS+=" --mel-fmax 22050.0"
#elif [ "$SAMPLING_RATE" == "16000" ]; then
#  ARGS+=" --sampling-rate 16000"
#  ARGS+=" --filter-length 1024"
#  ARGS+=" --hop-length 256"
#  ARGS+=" --win-length 1024"
#  ARGS+=" --mel-fmin 0.0"
#  ARGS+=" --mel-fmax 8000.0"
#elif [ "$SAMPLING_RATE" != "22050" ]; then
#  echo "Unknown sampling rate $SAMPLING_RATE"
#  exit 1
#fi

#  --dataset_config $DATASET_CONFIG \
ARGS+=" --train-dir $TRAIN_DIR   --dev-dir $DEV_DIR  --f0-stat $F0_STATS    --energy-stat $ENERGY_STATS    --dataset_mapping $DATASET_MAPPING  --dataset_stats $DATASET_STATS  "
ARGS+=" --use_char $USE_CHAR  --tones $TONES --toneseparated $TONESSEPARATED    --use-norm $USE_NORM --convert_ipa $CONVERT_IPA "
ARGS+=" --n-symbols $N_SYMBOLS  --padding-idx $PADDING_IDX  --symbols-embedding-dim  $SYMBOLS_EMBEDDING_DIM "
ARGS+=" --use_ipa_phone $USE_IPA_PHONE --load_language_array $LOAD_LANGUAGE_ARRAY --load_speaker_array $LOAD_SPEAKER_ARRAY --filter_on_lang $FILTER_ON_LANG "
ARGS+=" --filter_on_speaker $FILTER_ON_SPEAKER --filter_on_uttid $FILTER_ON_UTTID --mtype $MTYPE  --generated $GENERATED --train-type $TRAIN_TYPE --format  $FORMAT "
#ARGS+=" --variant  $VARIANT "
ARGS+=" --train_meta $TRAIN_META  --valid_meta $VALID_META  "
ARGS+=" --use-reversal-classifier 1  "
ARGS+=" --use-soft  "
#ARGS+=" --use-soft-tacotron  "
ARGS+=" --use-tgt-dur-soft  "
#ARGS+=" --use-mas  "
#ARGS+=" --use-diffusion  "
#ARGS+=" --use-priorgrad  "
ARGS+=" --train-wav  "
#ARGS+=" --use-sdtw  "
ARGS+=" --use-vq-gan "
ARGS+=" --train-vq "
#ARGS+=" --vq-use-gan-output-for-mel-generation "
#ARGS+=" --use-hifigan-wav-decoder "
#ARGS+=" --use-msmc-hifigan-wav-decoder "
#ARGS+=" --use-vq-frame-decoder "
#ARGS+=" --vq-pred-mel "
ARGS+=" --vq-use-fastpitch-transformer "

ARGS+=" --use-conformer "
#ARGS+=" --can-train-vq-only "
ARGS+=" --checkpoint-path  $CHECKPOINT_PATH "
ARGS+=" --filter_remove_lang_ids  $REMOVE_LANG_IDS "


mkdir -p "$OUTPUT_DIR"

: ${DISTRIBUTED:="-m torch.distributed.launch --nproc_per_node $NUM_GPUS"}
PYTHONPATH='.'  python $DISTRIBUTED train_me_char_wavvq.py $ARGS "$@"
