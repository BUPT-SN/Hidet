import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def build(opt):
    detail = pd.read_csv(opt.detail_file)
    if 'score' in detail.columns:
        selected = detail[detail['score'] > opt.threshold]['id'].tolist()
    else:
        selected = detail[detail['pred_label'] == 1]['id'].tolist()

    source = pd.read_csv(opt.source_file)
    filtered = source[source['id'].isin(selected)]
    if len(filtered) == 0:
        raise ValueError('No samples routed to the subdivision module; '
                         'check the detector predictions or the threshold.')

    os.makedirs(opt.output_dir, exist_ok=True)
    filtered.to_csv(os.path.join(opt.output_dir, 'valid.csv'), index=False)
    train_df, test_df = train_test_split(filtered, test_size=0.2, random_state=42)
    train_df.to_csv(os.path.join(opt.output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(opt.output_dir, 'test.csv'), index=False)
    print(f'{len(filtered)} samples routed to the subdivision module -> {opt.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detail_file', type=str, default='results/detail_result.csv')
    parser.add_argument('--source_file', type=str, default='data/seqxgpt/test.csv')
    parser.add_argument('--output_dir', type=str, default='data/middle')
    parser.add_argument('--threshold', type=float, default=0.35)
    opt = parser.parse_args()
    build(opt)
