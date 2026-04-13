# data_split.py

import torch
import random
from typing import Dict, Tuple, List
from torch.utils.data import SubsetRandomSampler, DataLoader
from random import sample

def split_dataset(datasets: Dict[str, torch.utils.data.Dataset], model_names: List[str], exp_mode: str, split_type='random') -> Tuple[Dict, Dict]:
    num = len(datasets[model_names[0]])
    indices = list(range(num))
    random.shuffle(indices)

    train_idx = indices[:int(0.9 * num)]
    valid_idx = indices[int(0.9 * num):]

    train_datasets = {m: datasets[m][torch.tensor(train_idx)] for m in model_names}
    valid_datasets = {m: datasets[m][torch.tensor(valid_idx)] for m in model_names}

    print(f"[→] Split dataset with train:val = {len(train_idx)}:{len(valid_idx)}")
    return train_datasets, valid_datasets

def split_dataset_by_fold(dataset, indices, fold, batch_size) -> Tuple[DataLoader, DataLoader, DataLoader]:
    fold_size = int(0.2 * len(indices))
    test_idx = indices[fold * fold_size:] if fold == 4 else indices[fold * fold_size:(fold + 1) * fold_size]
    train_idx = list(set(indices) - set(test_idx))
    valid_idx = sample(train_idx, int(0.1 * len(train_idx)))

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))

    return train_loader, valid_loader, test_loader
