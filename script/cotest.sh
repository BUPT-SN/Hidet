#!/bin/bash
# Two-stage inference of HiDet: coarse filtering -> subdivision detection -> merge.
set -e

COARSE_DIR=runs/coarse-seqxgpt_v0
SUBDIVISION_DIR=runs/subdivision-seqxgpt_v0

# Stage 1: coarse filtering.
python test.py \
    --module coarse \
    --model_dir ${COARSE_DIR} \
    --path data/seqxgpt \
    --split test \
    --threshold 0.35 \
    --detail_result_path results/detail_result.csv

# Build the middle dataset from samples judged as human by the coarse module.
python tool/build_middle.py \
    --detail_file results/detail_result.csv \
    --source_file data/seqxgpt/test.csv \
    --output_dir data/middle_test \
    --threshold 0.35

# Stage 2: subdivision detection on the routed samples.
python test.py \
    --module subdivision \
    --model_dir ${SUBDIVISION_DIR} \
    --path data/middle_test \
    --split valid \
    --threshold 0.5 \
    --detail_result_path results/second_detail_result.csv

# Merge the two stages into the final result.
python tool/merge_results.py \
    --file1 results/detail_result.csv \
    --file2 results/second_detail_result.csv \
    --output_file results/final_result.csv
