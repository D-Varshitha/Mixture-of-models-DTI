# data_split.py

import torch
import random
from typing import Dict, Tuple, List
from torch.utils.data import SubsetRandomSampler, DataLoader


def _resolve_collate_fn(dataset):
    # Try to find the collate_fn from the underlying dataset class or instance
    candidate = dataset
    if hasattr(candidate, 'dataset'): candidate = candidate.dataset # handle Subset
    
    res = getattr(candidate, 'collate_fn', None) or getattr(candidate, 'moe_collate_fn', None)
    if res: return res
    
    # Fallback for MoEDataset specifically
    try:
        from data.moe_dataset import moe_collate_fn
        return moe_collate_fn
    except ImportError:
        return None

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
    num_folds = 5
    fold_size = len(indices) // num_folds
    start = fold * fold_size
    end = len(indices) if fold == (num_folds - 1) else (fold + 1) * fold_size

    test_idx = indices[start:end]
    train_candidates = indices[:start] + indices[end:]

    valid_size = max(1, int(0.1 * len(train_candidates)))
    valid_idx = train_candidates[:valid_size]
    train_idx = train_candidates[valid_size:]
    if len(train_idx) == 0:
        raise RuntimeError("[CV Split] Training split became empty after validation split.")

    collate_fn = _resolve_collate_fn(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx), collate_fn=collate_fn)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx), collate_fn=collate_fn)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx), collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader
