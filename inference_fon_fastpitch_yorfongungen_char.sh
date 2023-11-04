#!/usr/bin/env bash

#: ${WAVEGLOW:="/home/phifamin/benit/FastPitchMe/pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
#: ${WAVEGLOW:="/home/phifamin/benit/vocoder/waveglow_pytorch_me/waveglowcheckpoints/waveglow_214000"}
: ${WAVEGLOW:="SKIP"}
#: ${HIFIGAN:="/home/phifamin/benit/FastPitchMe/pretrained_models/hifigan/g_00750000"}
: ${HIFIGAN:="SKIP"}
: ${HIFIGAN2:="/media/phifamin/DATA/mebenit/test2/trainhifigan/model-890000.pt"}
: ${PRIORGRAD:="/media/phifamin/DATA/mebenit/test2/priorgradtrain/weights-500000.pt"}
#: ${HIFIGAN2:="SKIP"}
: ${MELGAN:="/media/phifamin/DATA/mebenit/test2/trainmelgan_neurips/best_netG.pt"}
: ${FASTPITCH:="/media/phifamin/DATA/mebenit/test2/trainoutput/FastPitchMulti_checkpoint_115.pt"}
: ${BATCH_SIZE:=1}
#: ${PHRASES:="phrases/devset10.tsv"}
: ${OUTPUT_DIR:="/media/phifamin/DATA/mebenit/test2/fastpitchtest/"}
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${SEED:=""}
: ${LEARNING_RATE:=0.1}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}

: ${SPEAKER:=-1}
#: ${NUM_SPEAKERS:=1}


echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"
: ${SAMPLING_RATE:=16000}


#: ${TRAIN_DIR:=/scratch/roseline/dump/DumpFongbefrall/train/}
#: ${DEV_DIR:=/scratch/roseline/dump/DumpFongbefrall/valid/}
#: ${F0_STATS:=/scratch/roseline/dump/DumpFongbefrall/stats_f0.npy}
#: ${ENERGY_STATS:=/scratch/roseline/dump/DumpFongbefrall/stats_energy.npy}
#: ${DATASET_MAPPING:=/scratch/roseline/dump/DumpFongbefrall/mapper_fonfr_char.json}
##: ${DATASET_CONFIG:=/scratch/roseline/dump/TensorFlowTTS/preprocess/preprocess_francaisfongbesmultispeakers.yaml }
#: ${DATASET_STATS:=/scratch/roseline/dump/DumpFongbefrall/stats.npy}
: ${TRAIN_DIR:=/media/phifamin/DATA/mebenit/test2/}
: ${DEV_DIR:=/media/phifamin/DATA/mebenit/test2/vals/}
: ${F0_STATS:=/home/roseline/benit/DumpFongbefrall/stats_f0.npy}
: ${ENERGY_STATS:=/home/roseline/benit/DumpFongbefrall/stats_energy.npy}
#: ${DATASET_MAPPING:=/home/roseline/benit/DumpFongbefrall/fr_fon_mapper.json}
: ${DATASET_MAPPING:=/media/phifamin/DATA/mebenit/test2/mapper_fon_yor_gun_gen_char.json}
#: ${DATASET_CONFIG:=/home/roseline/benit/TensorFlowTTS/preprocess/preprocess_francaisfongbesmultispeakers.yaml }
: ${DATASET_STATS:=/home/roseline/benit/DumpFongbefrall/stats.npy}

#: ${TRAIN_META:=/media/phifamin/DATA/mebenit/test2/metadata_fon_gun_yor.csv}
: ${VALID_META:=/media/phifamin/DATA/mebenit/test2/metadata_fon_gun_yor_mina_cmu_valid.csv}

: ${DATASET_PATH:=/media/phifamin/DATA/mebenit/test2/vals/}
#: ${F0_STATS:=/home/roseline/benit/DumpFongbefrall/stats_f0.npy}
#: ${ENERGY_STATS:=/home/roseline/benit/DumpFongbefrall/stats_energy.npy}
#: ${DATASET_MAPPING:=/home/roseline/benit/DumpFongbefrall/fr_fon_mapper.json}
#: ${DATASET_CONFIG:=/home/roseline/benit/TensorFlowTTS/preprocess/preprocess_francaisfongbesmultispeakers.yaml }
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
: ${REMOVE_LANG_IDS:=english,french,yoruba}

: ${MTYPE:=2}

: ${GENERATED:=0}
#: ${TRAIN_TYPE:=fastpitch}
: ${TRAIN_TYPE:=create}
: ${N_MEL_CHANNELS:=80}
: ${FORMAT:=pt}
: ${N_SYMBOLS:=300}
: ${PADDING_IDX:=0}
: ${SYMBOLS_EMBEDDING_DIM:=384}


#: ${VARIANT:=FASTSPEECHMODEL}





ARGS=""
#ARGS+=" -i $PHRASES "
#ARGS+=" --reversal-classifier "
#ARGS+="  --use-reversal-classifier 1 "
ARGS+="  --use-reversal-classifier 1 "
#ARGS+=" --cuda "
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --fastpitch $FASTPITCH"
ARGS+=" --waveglow $WAVEGLOW"
ARGS+=" --hifigan $HIFIGAN"
ARGS+=" --hifigan2 $HIFIGAN2"
ARGS+=" --priorgrad $PRIORGRAD"
ARGS+=" --melgan $MELGAN"
ARGS+=" --wn-channels 256"
ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --text-cleaners english_cleaners_v2"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --repeats $REPEATS"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --speaker $SPEAKER"
ARGS+=" --save-mels"
#ARGS+=" -lr $LEARNING_RATE"
#ARGS+=" --weight-decay 1e-8"
#ARGS+=" --grad-clip-thresh 1000.0"
#ARGS+=" --dur-predictor-loss-scale 0.1"
#ARGS+=" --pitch-predictor-loss-scale 0.1"
#ARGS+=" --grad-accumulation $GRAD_ACCUMULATION"

# Autoalign & new features
#ARGS+=" --kl-loss-start-epoch 0"
#ARGS+=" --kl-loss-warmup-epochs $KL_LOSS_WARMUP"


#ARGS+=" --n-speakers $NUM_SPEAKERS"
[ "$CPU" = false ]          && ARGS+=" --cuda"
[ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
[ "$AMP" = true ]           && ARGS+=" --amp"
[ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
[ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"
#  --dataset_config $DATASET_CONFIG \
#ARGS+=" --train-dir $TRAIN_DIR   --dev-dir $DEV_DIR "
#
ARGS+="   --f0-stat $F0_STATS    --energy-stat $ENERGY_STATS    --dataset_mapping $DATASET_MAPPING  --dataset_stats $DATASET_STATS  "
ARGS+=" --use_char $USE_CHAR  --tones $TONES --toneseparated $TONESSEPARATED    --use-norm $USE_NORM --convert_ipa $CONVERT_IPA "

ARGS+=" --nmel-channels $N_MEL_CHANNELS"
[ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 1.0"

ARGS+=" --n-symbols $N_SYMBOLS --padding-idx $PADDING_IDX  --symbols-embedding-dim  $SYMBOLS_EMBEDDING_DIM "

#ARGS+=" --n-symbolss $N_SYMBOLS --padding-idxs $PADDING_IDX  --symbols-embedding-dims  $SYMBOLS_EMBEDDING_DIM "


ARGS+=" --use_ipa_phone $USE_IPA_PHONE --load_language_array $LOAD_LANGUAGE_ARRAY --load_speaker_array $LOAD_SPEAKER_ARRAY --filter_on_lang $FILTER_ON_LANG "
ARGS+=" --filter_on_speaker $FILTER_ON_SPEAKER --filter_on_uttid $FILTER_ON_UTTID --mtype $MTYPE  --generated $GENERATED --train-type $TRAIN_TYPE --format  $FORMAT "
ARGS+=" --dataset-path $DATASET_PATH  "
#ARGS+=" --train_meta $TRAIN_META  --valid_meta $VALID_META  "
ARGS+="   --valid_meta $VALID_META  "

ARGS+=" --filter_remove_lang_ids  $REMOVE_LANG_IDS "
#ARGS+=" --variant  $VARIANT "


mkdir -p "$OUTPUT_DIR"
#echo -e "\nargs=\n$ARGS\n"

python inference_me_char.py $ARGS "$@"
