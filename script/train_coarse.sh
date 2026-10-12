#!/bin/bash
# Train the coarse module (Task 1 data: human vs. original AI texts).
set -e

python train.py \
    --module coarse \
    --device_num 1 \
    --path data/seqxgpt \
    --database_name train \
    --test_dataset_name valid \
    --name coarse-seqxgpt \
    --total_epoch 30 \
    --lr 2e-5 \
    --warmup_steps 2000 \
    --per_gpu_batch_size 32 \
    --per_gpu_eval_batch_size 32
