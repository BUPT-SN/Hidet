import os

import pandas as pd

METHOD_TAXONOMY = {
    'char': ['spaceremoval', 'insertspace', 'wordcase_pct50', 'wordmerge',
             'punctuationremoval', 'punctuationappend'],
    'word': ['typos', 'spellingerror', 'insertadv', 'mlmword'],
    'sent': ['appendirr', 'backtransent', 'repeatsent', 'mlmsent'],
    'para': ['reverse', 'backtran', 'paraphrase'],
    'human': ['human'],
    'origin': ['machine_origin'],
}

LEVEL_IDS = {'char': 0, 'word': 1, 'sent': 2, 'para': 3, 'human': 4, 'origin': 5}

HUMANIZED_LEVELS = (0, 1, 2, 3)


def build_method_index():
    index = {}
    method_id = 0
    for level, methods in METHOD_TAXONOMY.items():
        for name in methods:
            index[name] = (method_id, LEVEL_IDS[level])
            method_id += 1
    return index


METHOD_INDEX = build_method_index()


def trim_quotes(s):
    return s.strip("\"'")


def process_spaces(text):
    text = text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()
    return trim_quotes(text)


def load_MyData(file_folder):
    """Load train/valid/test.csv splits from a folder.

    Each CSV needs columns: id, Generation, label. The label is 'human',
    'machine_origin' or the name of a humanizing method (see METHOD_TAXONOMY).
    Returns {'train': [(text, label, src, id), ...], 'valid': [...], 'test': [...]}
    where label is '1' (human) or '0' (AI).
    """
    data = {'train': [], 'test': [], 'valid': []}
    for now in sorted(os.listdir(file_folder)):
        if not now.endswith('.csv'):
            continue
        keyname = now.split('.')[0]
        assert keyname in data, f'{keyname} is not in {list(data.keys())}'
        df = pd.read_csv(os.path.join(file_folder, now), on_bad_lines='skip')
        for i in range(len(df)):
            row = df.iloc[i]
            text, src, sid = row['Generation'], row['label'], row['id']
            label = '1' if src == 'human' else '0'
            data[keyname].append((process_spaces(str(text)), label, src, sid))
    return data
