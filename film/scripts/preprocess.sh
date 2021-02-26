#!/usr/bin/env bash
questions_dirname=$1

python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/$questions_dirname/CLEVR_train_questions.json \
  --output_h5_file data/CLEVR_v1.0/$questions_dirname/train_questions.h5 \
  --output_vocab_json data/CLEVR_v1.0/$questions_dirname/vocab.json

python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/$questions_dirname/CLEVR_val_questions.json \
  --output_h5_file data/CLEVR_v1.0/$questions_dirname/val_questions.h5 \
  --input_vocab_json data/CLEVR_v1.0/$questions_dirname/vocab.json
