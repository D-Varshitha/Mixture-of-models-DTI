import torch
import os
import sys
import random
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None
from torch import nn
from tqdm import trange

from data import split_dataset_by_fold
from data.moe_dataset import MoEDataset
from models import build_model
from models.model_MoE import DTI_Sparse_MoE
from engine.trainer_MoE import train_moe, test_moe
from config import args, metrics_classification, metrics_regression


def _validate_dataset_integrity(dataset):
    """FIX 2: Run once before training to ensure all required keys and shapes are present."""
    required_keys = [
        'label', 'shared_drug', 'shared_prot', 'shared_drug_mask', 'shared_prot_mask',
        'dpdta_com', 'dpdta_pro',
        'dcdti_com', 'dcdti_pro',
        'mdprd_com', 'mdprd_pro',
        'gifdti_com', 'gifdti_pro', 'gifdti_com_mask', 'gifdti_pro_mask',
        'pcpi_graph', 'pcpi_morgan', 'pcpi_sequence',
        'dp_af', 'dp_bf', 'dp_ag', 'dp_bg', 'dp_abn', 'dp_pro',
    ]
    base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    sample_ids = sorted(set([0, max(0, len(base) // 2), max(0, len(base) - 1)]))
    for sample_id in sample_ids:
        sample = base[sample_id]

        missing = [k for k in required_keys if k not in sample]
        if missing:
            raise RuntimeError(f"[Dataset Integrity] Missing keys in dataset sample {sample_id}: {missing}")

        empty = []
        for k, v in sample.items():
            if isinstance(v, torch.Tensor) and v.numel() == 0:
                empty.append(k)
            elif isinstance(v, list) and len(v) == 0:
                empty.append(k)
        if empty:
            raise RuntimeError(f"[Dataset Integrity] Empty tensors in sample {sample_id}: {empty}")

        if sample['dcdti_com'].shape[-1] != 2048:
            raise RuntimeError(f"[Dataset Integrity] dcdti_com dim mismatch in sample {sample_id}: {sample['dcdti_com'].shape}")
        if sample['mdprd_com'].shape[-1] != 1024:
            raise RuntimeError(f"[Dataset Integrity] mdprd_com dim mismatch in sample {sample_id}: {sample['mdprd_com'].shape}")
        if sample['pcpi_morgan'].shape[-1] != 1024:
            raise RuntimeError(f"[Dataset Integrity] pcpi_morgan dim mismatch in sample {sample_id}: {sample['pcpi_morgan'].shape}")
        if tuple(sample['mdprd_pro'].shape) != (5, 500, 500):
            raise RuntimeError(f"[Dataset Integrity] mdprd_pro shape mismatch in sample {sample_id}: {sample['mdprd_pro'].shape}")
        if sample['shared_drug'].ndim != 2 or sample['shared_prot'].ndim != 2:
            raise RuntimeError(f"[Dataset Integrity] Shared embeddings must stay token-level in sample {sample_id}.")
        if sample['dcdti_pro'].dtype != torch.long:
            raise RuntimeError(f"[Dataset Integrity] dcdti_pro must be integer token IDs in sample {sample_id}.")

    print(f"[Dataset Integrity] All {len(required_keys)} required keys present. ✓")


def _build_experts_and_model(args, device):
    """FIX 4: Called inside the fold loop to reset model state per fold."""
    experts_dict = {
        'dpdta':        build_model('dpdta',        args.task),
        'dcdti':        build_model('dcdti',        args.task),
        'dp':           build_model('dp',           args.task),
        'mdprd':        build_model('mdprd',        args.task),
        'gifdti':       build_model('gifdti',       args.task),
        'perceivercpi': build_model('perceivercpi', args.task),
    }
    moe_model = DTI_Sparse_MoE(
        experts_dict, drug_vocab=65, prot_vocab=26, k=2, lambda_aux=args.lambda_aux
    ).to(device)
    optimizer = torch.optim.Adam(
        moe_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    return moe_model, optimizer


def main():

    # ── Hardware detection ────────────────────────────────────────────────────
    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device('cuda')
        print("Hardware detected: GPU")
    else:
        device = torch.device('cpu')
        print("Hardware detected: CPU. This is great for local testing!")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_name    = args.data if args.data else 'kiba'
    label_col    = 'lab' if args.task == 'classification' else 'affinity'
    args.label   = label_col   # keep args.label in sync so trainer metrics use correct column
    dataset_root = os.getcwd()

    # ── FIX 2: Dataset loading ────────────────────────────────────────────────
    # Logic: In debug mode, we only process a small subset for extreme speed.
    current_subset = args.subset_size if args.mode == 'debug' else None
    print(f"Loading Dataset: {data_name} (mode={args.get_dataset}, subset={current_subset})...")

    if args.get_dataset == 'generate':
        _ = MoEDataset(
            root=dataset_root, dataset_name=data_name,
            label_type=label_col, mode='generate', subset_size=current_subset,
            MAX_SMI_LEN=args.com_len, MAX_SEQ_LEN=args.pro_len,
        )
        print("Feature generation done. Reloading dataset...")

    dataset = MoEDataset(
        root=dataset_root, dataset_name=data_name,
        label_type=label_col, mode='load', subset_size=current_subset,
        MAX_SMI_LEN=args.com_len, MAX_SEQ_LEN=args.pro_len,
    )

    # ── Debug mode adjustments ────────────────────────────────────────────────
    if args.mode == 'debug':
        print(f"DEBUG MODE: Subset of {len(dataset)} items active. Restricting to 1 FOLD.")
        args.epoch = min(args.epoch, 3)
        args.batch = min(args.batch, 16)
        args.exp_mode = 'pure_train'  # Restrict to 1 fold


    all_idx = list(range(len(dataset)))
    random.Random(args.seed).shuffle(all_idx)

    # ── FIX 2: Dataset integrity check ────────────────────────────────────────
    _validate_dataset_integrity(dataset)

    # ── Loss function ─────────────────────────────────────────────────────────
    if args.task == 'classification':
        loss_fn     = nn.BCEWithLogitsLoss()
        args.metrics = metrics_classification
    else:
        loss_fn     = nn.MSELoss()
        args.metrics = metrics_regression

    # ── K-Fold execution pipeline ─────────────────────────────────────────────
    print(f"Execution Pipeline starting. Mode={args.mode}")
    folds_to_run = 5 if args.exp_mode == '5_fold_val' else 1

    for fold in range(folds_to_run):
        print(f"\n===== FOLD {fold} =====")

        # ── FIX 4: Re-initialize model + optimizer per fold for valid CV ──────
        print("Building MoE Model (fresh initialisation for this fold)...")
        moe_model, optimizer = _build_experts_and_model(args, device)

        from data.data_split import split_dataset_by_fold
        train_loader, valid_loader, test_loader = split_dataset_by_fold(
            dataset, all_idx, fold, args.batch
        )

        if wandb is not None:
            wandb.init(
                project='ensdti-moe',
                name=f'MoE_{data_name}_fold{fold}',
                config=args, reinit=True
            )

        # Train
        moe_model = train_moe(
            train_loader, moe_model, loss_fn, optimizer, args,
            valid_loader=valid_loader
        )

        # Test
        print(f"\nFinal Testing for FOLD {fold}")
        result, perf = test_moe(test_loader, moe_model, loss_fn, args, split='Test_Final')

        if args.save_result:
            out_path = (f'{os.getcwd()}/dataset/{data_name}/moe/'
                        f'{args.exp_mode}/{args.custom}/fold_{fold}')
            os.makedirs(out_path, exist_ok=True)
            result.to_csv(f'{out_path}/test_result.csv', index=False)
            if args.save_perf:
                perf.to_csv(f'{out_path}/test_perf.csv', index=False)
        if wandb is not None and wandb.run is not None:
            wandb.finish()

if __name__ == '__main__':
    main()
