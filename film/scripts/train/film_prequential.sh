#!/bin/bash

TN=$1
MN=$2
BS=$3
LR=$4
NTE=$5
SEED=$6
PBN=$7

if [[ $TN == *"clevr.comparison"* ]]; then
    PMBS=83105
elif [[ $TN == *"clevr.compare_integer"* ]]; then
    PMBS=15942
elif [[ $TN == *"clevr.same_relate"* ]]; then
    PMBS=82604
else
    echo "PMBS not set for TN = $TN"
    exit 0
fi
echo "PMBS=$PMBS"

OUTPUT_DIR="exp/rda/tn-$TN.mn-$MN.bs-$BS.lr-$LR.nte-$NTE.seed-$SEED.pbn-$PBN"
mkdir -p $OUTPUT_DIR
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SLURM_NODELIST=$SLURM_NODELIST"
DATA_DIR="data/CLEVR_v1.0/$TN"
echo "DATA_DIR=$DATA_DIR"
if [[ $PBN -eq 8 ]]; then
    SPLIT="val"
else
    SPLIT="train"
fi
echo "SPLIT=$SPLIT for feature loading"

python scripts/train_model.py \
  --checkpoint_path "$OUTPUT_DIR/model.pt" \
  --train_question_h5 "data/CLEVR_v1.0/$TN/train_questions.h5" \
  --val_question_h5 "data/CLEVR_v1.0/$TN/val_questions.h5" \
  --vocab_json "data/CLEVR_v1.0/$TN/vocab.json" \
  --model_type $MN \
  --patience_epochs 2 \
  --num_train_epochs $NTE \
  --prequential_block_no $PBN \
  --prequential_max_block_size $PMBS \
  --print_verbose_every 20000000 \
  --stopping_criterion loss \
  --shuffle_train_data $SEED \
  --optimizer Adam \
  --learning_rate $LR \
  --batch_size $BS \
  --use_coords 1 \
  --module_stem_batchnorm 1 \
  --module_stem_num_layers 1 \
  --module_batchnorm 1 \
  --classifier_batchnorm 1 \
  --bidirectional 0 \
  --decoder_type linear \
  --encoder_type gru \
  --weight_decay 1e-5 \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 200 \
  --rnn_hidden_dim 4096 \
  --rnn_output_batchnorm 0 \
  --classifier_downsample maxpoolfull \
  --classifier_proj_dim 512 \
  --classifier_fc_dims 1024 \
  --module_input_proj 1 \
  --module_residual 1 \
  --module_dim 128 \
  --module_dropout 0e-2 \
  --module_stem_kernel_size 3 \
  --module_kernel_size 3 \
  --module_batchnorm_affine 0 \
  --module_num_layers 1 \
  --num_modules 4 \
  --condition_pattern 1,1,1,1 \
  --gamma_option linear \
  --gamma_baseline 1 \
  --use_gamma 1 \
  --use_beta 1 \
  --condition_method bn-film \
  --program_generator_parameter_efficient 1

for PS in "val" "test"; do
    python scripts/run_model.py \
      --program_generator $OUTPUT_DIR/model.pt \
      --execution_engine $OUTPUT_DIR/model.pt \
      --input_question_h5 "$DATA_DIR/${SPLIT}_questions.h5" \
      --input_features_h5 "data/${SPLIT}_features.h5" \
      --output_preds "$OUTPUT_DIR/${PS}_preds.txt" \
      --output_h5 "$OUTPUT_DIR/${PS}_preds.h5" \
      --shuffle_train_data $SEED \
      --prequential_block_no $PBN \
      --prequential_max_block_size $PMBS \
      --prequential_split $PS \
      --use_gpu 1
done
