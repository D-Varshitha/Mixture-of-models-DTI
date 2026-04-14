# data_loader.py

import os
import torch
from .dataset import CustomCPIDataset  # 假设你已实现

def generate_dataset_by_model(data_name: str, model_name: str, args) -> torch.utils.data.Dataset:
    dataset_path = os.path.join(os.getcwd(), 'dataset', data_name, model_name, 'dataset.pt')
    
    if os.path.exists(dataset_path):
        print(f"[✓] Loading existing dataset from {dataset_path}")
        return torch.load(dataset_path)

    print(f"[+] Generating dataset for model: {model_name}")
    if model_name == 'dcdti':
        return CustomCPIDataset(os.getcwd(), data_name, 100, 2500, args.label, 'fp_2048', 'seq_enc', args.get_dataset)
    elif model_name == 'dpdta':
        com_len, pro_len = 100, 1000
        if data_name == 'davis':
            com_len, pro_len = 85, 1200
        return CustomCPIDataset(os.getcwd(), data_name, com_len, pro_len, args.label, 'smi_enc', 'seq_enc', args.get_dataset)
    elif model_name == 'mdprd':
        return CustomCPIDataset(os.getcwd(), data_name, None, None, args.label, 'fp_1024', 'mdprd', args.get_dataset)
    elif model_name == 'dp':
        return CustomCPIDataset(os.getcwd(), data_name, args.com_len, args.pro_len, args.label, 'mpnn', 'dp', args.get_dataset)
    elif model_name == 'cpi':
        return CustomCPIDataset(os.getcwd(), data_name, None, None, args.label, 'subgraph', 'word', args.get_dataset)
    elif model_name == 'ensdti':
        return CustomCPIDataset()
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
