#!/bin/bash
# Train the subdivision module on the samples routed by the coarse module.
# Requires a trained coarse model (script/train_coarse.sh) and updates the run
# directory below accordingly.
set -e

COARSE_DIR=runs/coarse-seqxgpt_v0

# Route the training split through the coarse module (threshold 0.35, Eq. 2).
python test.py \
    --module coarse \
    --model_dir ${COARSE_DIR} \
    --path data/seqxgpt \
    --split train \
    --threshold 0.35 \
    --detail_result_path results/coarse_train_detail.csv

# Build the middle dataset from the routed samples.
python tool/build_middle.py \
    --detail_file results/coarse_train_detail.csv \
    --source_file data/seqxgpt/train.csv \
    --output_dir data/middle \
    --threshold 0.35

# Train the subdivision module with multi-grained contrastive learning.
python train.py \
    --module subdivision \
    --device_num 1 \
    --path data/middle \
    --database_name train \
    --test_dataset_name test \
    --name subdivision-seqxgpt \
    --total_epoch 15 \
    --lr 1e-5 \
    --warmup_steps 1000 \
    --per_gpu_batch_size 8 \
    --per_gpu_eval_batch_size 32 \
    --freeze_embedding_layer
