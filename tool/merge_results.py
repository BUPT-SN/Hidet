import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import compute_metrics, print_metrics


def merge(opt):
    df1 = pd.read_csv(opt.file1)
    df2 = pd.read_csv(opt.file2)
    merged = pd.concat([df2, df1[~df1['id'].isin(df2['id'])]], ignore_index=True)
    merged.sort_values(by='id', inplace=True)
    os.makedirs(os.path.dirname(opt.output_file) or '.', exist_ok=True)
    merged.to_csv(opt.output_file, index=False)

    scores = merged['score'].tolist() if 'score' in merged.columns else None
    m = compute_metrics(merged['true_label'].tolist(), merged['pred_label'].tolist(),
                        scores, merged['id'].tolist())
    print('\nFinal two-stage metrics:')
    print_metrics(m)
    print(f'Saved final results to {opt.output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='results/detail_result.csv')
    parser.add_argument('--file2', type=str, default='results/second_detail_result.csv')
    parser.add_argument('--output_file', type=str, default='results/final_result.csv')
    opt = parser.parse_args()
    merge(opt)
