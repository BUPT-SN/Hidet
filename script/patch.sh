#!/bin/bash
# Patch an existing detector with the HiDet subdivision module.
#
# Prepare results/detail_result.csv with your detector's predictions on
# data/seqxgpt/test.csv first: columns id, true_label, pred_label and
# optionally score = P(human). Subdivision module trained via
# script/train_subdivision.sh (ideally retrained on middle data built from
# your detector's predictions on the training split).
set -e

SUBDIVISION_DIR=runs/subdivision-seqxgpt_v0

python tool/build_middle.py \
    --detail_file results/detail_result.csv \
    --source_file data/seqxgpt/test.csv \
    --output_dir data/middle_test

python test.py \
    --module subdivision \
    --model_dir ${SUBDIVISION_DIR} \
    --path data/middle_test \
    --split valid \
    --threshold 0.5 \
    --detail_result_path results/second_detail_result.csv

python tool/merge_results.py \
    --file1 results/detail_result.csv \
    --file2 results/second_detail_result.csv \
    --output_file results/final_result.csv
