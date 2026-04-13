# data_utils.py

import torch
import random
from typing import Dict, List, Union, Tuple
from data_loader import generate_dataset_by_model
from data_split import split_dataset, split_dataset_by_fold

def load_data(data_name: str, model_names: List[str], args) -> Dict[str, torch.utils.data.Dataset]:
    datasets = {}
    for m in model_names:
        datasets[m] = generate_dataset_by_model(data_name, m, args)
    return datasets

def return_dataloader(dataset: torch.utils.data.Dataset, batch: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle)

def prepare_data(data: Union[str, List[str]], model_names: List[str], args):
    if isinstance(data, list) and len(data) == 2:
        train_data, test_data = data
        train_datasets = load_data(train_data, model_names, args)
        test_datasets = load_data(test_data, model_names, args)
        train_datasets, valid_datasets = split_dataset(train_datasets, model_names, args.exp_mode, args.split_type)
        return train_datasets, valid_datasets, test_datasets

    datasets = load_data(data, model_names, args)

    if args.exp_mode == '5_fold_val':
        indices = list(range(len(datasets[model_names[0]])))
        random.shuffle(indices)
        return datasets, indices
    elif args.exp_mode in ['pure_train', 'train_and_test']:
        return split_dataset(datasets, model_names, args.exp_mode, args.split_type)
    else:
        return datasets
