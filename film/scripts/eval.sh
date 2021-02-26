#!/bin/bash

checkpoint_dir=${@}
python scripts/run_model.py \
  --program_generator $checkpoint_dir/model.pt \
  --execution_engine $checkpoint_dir/model.pt \
  --input_question_h5 data/val_questions.h5 \
  --input_features_h5 data/val_features.h5 \
  --output_preds $checkpoint_dir/val_preds.txt \
  --output_h5 $checkpoint_dir/val_preds.h5

python CLEVR_eval_with_q_type.py \
  --questions_file data/CLEVR_v1.0/questions/CLEVR_val_questions.json \
  --answers_file $checkpoint_dir/val_preds.txt
