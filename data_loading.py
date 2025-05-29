import os
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_dataset(csv_path: os.PathLike, class_thresh: int = 100):
    CLEAN_NAMES = {
        '% GFP expression -1 PRF reporter (of maximal GFP expression measured in the assay)': "gfp_minus_rate",
        '% GFP expression +1 PRF reporter (of maximal GFP expression measured in the assay)': "gfp_plus_rate",
        'sequence tested': 'sequence',
        'variant id': 'variant_id',
        'PRF event': 'prf_class'
    }

    df = pd.read_csv(csv_path)
    df = df.rename(columns=CLEAN_NAMES)
    df = df[list(CLEAN_NAMES.values())]
    df = df.dropna(subset=['gfp_minus_rate', 'gfp_plus_rate'], thresh=1)
    df = df.dropna(subset=['prf_class'])

    class_counts = df['prf_class'].value_counts()
    df = df[df['prf_class'].map(lambda x: class_counts[x] > class_thresh)]
    df['gfp_minus'] = df['gfp_minus_rate'] > 1.3
    df['gfp_plus'] = df['gfp_plus_rate'] > 1.3

    return df


def encode_seq(seq: str) -> np.ndarray:
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    
    X = np.zeros((4, len(seq)))
    for i, c in enumerate(seq):
        X[nuc_to_idx[c], i] = 1

    return X
    

def build_tensors(data: pd.DataFrame, key: Literal['minus', 'plus'], size: int = 162, stride: int = 1):
    X_acc = []
    y_class_acc = []
    y_reg_acc = []

    for _, row in data.iterrows():
        seq = row['sequence']
        for end in range(size, len(seq)+1, stride):
            X_acc.append(encode_seq(seq[end-size:end]))
            y_class_acc.append(row[f'gfp_{key}'])
            y_reg_acc.append(row[f'gfp_{key}_rate'])

    X = torch.FloatTensor(np.stack(X_acc))
    y_class = torch.FloatTensor(y_class_acc)
    y_reg = torch.FloatTensor(y_reg_acc)
    
    return X, y_class, y_reg


def build_dataloaders(data: pd.DataFrame,
                      key: Literal['minus', 'plus'],
                      test_split: Union[str, Tuple[str, str]] = 'mix',
                      test_size: float = 0.2,
                      batch_size: int = 32,
                      data_augment: bool = False):
    _LABEL = {
        'minus': ['gfp_minus_rate'],
        'plus': ['gfp_plus_rate'],
    }
    data = data.dropna(subset=f'gfp_{key}_rate')

    # Split data
    if test_split == 'mix':
        train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    elif isinstance(test_split, str):
        if test_split in data['prf_class'].unique():
            train_data = data[data['prf_class'] != test_split]
            val_data = data[data['prf_class'] == test_split]
        else:
            raise ValueError(f'The `test_split` argument {test_split} must be either mix or a valid prf_class')
    elif isinstance(test_split, tuple) and len(test_split) == 2:
        if all(x in data['prf_class'].unique() for x in test_split):
            if test_split[0] != test_split[1]:
                train_data = data[data['prf_class'] == test_split[0]]
                val_data = data[data['prf_class'] == test_split[1]]
            else:
                train_data, val_data = train_test_split(data[data['prf_class'] == test_split[0]], test_size=test_size, random_state=42)
        else:
            raise ValueError(f"One or more values in `test_split` {test_split} are not valid prf_class values")

    if data_augment:
        X_train, y_class_train, y_reg_train = build_tensors(train_data, key, size=150, stride=3)
        tmp = val_data.copy()
        tmp['sequence'] = tmp['sequence'].map(lambda x: x[6:-6])
        X_val, y_class_val, y_reg_val = build_tensors(tmp, key, 150, 1)
    else:
        X_train, y_class_train, y_reg_train = build_tensors(train_data, key)
        X_val, y_class_val, y_reg_val = build_tensors(val_data, key)


    # Create datasets and dataloaders using TensorDataset
    train_dataset = TensorDataset(X_train, y_class_train, y_reg_train)
    val_dataset = TensorDataset(X_val, y_class_val, y_reg_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader