import torch
import os
import sys
import copy
import random
import numpy as np
import time
import json
import pandas as pd

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
from torch import nn
from torch.nn.parallel import DataParallel

class CustomDataParallel(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        batch = inputs[0]
        B = batch['label'].shape[0]
        num_gpus = len(device_ids)
        chunk_size = (B + num_gpus - 1) // num_gpus

        scattered_batches = []
        for i in range(num_gpus):
            start = i * chunk_size
            end = min(start + chunk_size, B)
            if start >= B:
                break

            sub_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    sub_batch[k] = v[start:end].to(device_ids[i])
                elif isinstance(v, (list, tuple)):
                    sub_batch[k] = v[start:end]
                else:
                    sub_batch[k] = v
            scattered_batches.append((sub_batch,))

        scattered_kwargs = [{}] * len(scattered_batches)
        return scattered_batches, scattered_kwargs

from tqdm import trange

from data.moe_dataset import MoEDataset, moe_collate_fn
from models import build_model
from models.model_MoE import DTI_Sparse_MoE
from engine.trainer_MoE import train_moe, test_moe
from engine.conformal import get_calibration_scores, apply_icp_reference_logic
from engine.metrics import calculate_icp_selective_metrics
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


def _build_experts_and_model(args, device, dataset):
    """FIX 4: Called inside the fold loop to reset model state per fold."""
    experts_dict = {
        'dpdta':        build_model('dpdta',        args.task, com_len=args.com_len, pro_len=args.pro_len),
        'dcdti':        build_model('dcdti',        args.task, com_len=args.com_len, pro_len=args.pro_len),
        'dp':           build_model('dp',           args.task, com_len=args.com_len, pro_len=args.pro_len),
        'mdprd':        build_model('mdprd',        args.task, com_len=args.com_len, pro_len=args.pro_len),
        'gifdti':       build_model('gifdti',       args.task, com_len=args.com_len, pro_len=args.pro_len),
        'perceivercpi': build_model('perceivercpi', args.task, com_len=args.com_len, pro_len=args.pro_len),
    }
    
    # Extract actual dimensions from the embedding generator
    drug_dim = dataset.generator.chembert_hidden_size
    prot_dim = dataset.generator.esm_hidden_size
    
    moe_model = DTI_Sparse_MoE(
        experts_dict, drug_vocab=65, prot_vocab=26, k=args.top_k, 
        lambda_aux=args.lambda_aux,
        drug_pretrained_dim=drug_dim,
        prot_pretrained_dim=prot_dim
    ).to(device)
    
    if torch.cuda.device_count() > 1 and args.device != 'cpu':
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        moe_model = CustomDataParallel(moe_model)
        
    optimizer = torch.optim.Adam(
        moe_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    return moe_model, optimizer


def main():
    start_all = time.time()
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

    data_name    = args.data if args.data else 'davis'
    label_col    = 'lab' if args.task == 'classification' else 'affinity'
    args.label   = label_col
    dataset_root = args.root if args.root else os.getcwd()

    print(f"\nLoading dataset: {data_name}  |  task: {args.task}")
    print(f"  Embeddings  : ESM + ChemBERT — generated dynamically per sample, RAM-cached only")
    print(f"  Expert feats: fingerprints / MPNN / MDeePred — auto-generated on disk if missing")
    print(f"  Saved to disk: only the single best model weights\n")

    current_subset = args.subset_size if args.mode == 'debug' else None

    dataset = MoEDataset(
        root=dataset_root, dataset_name=data_name,
        label_type=label_col, subset_size=current_subset,
        MAX_SMI_LEN=args.com_len, MAX_SEQ_LEN=args.pro_len, mode=args.get_dataset,
    )

    _validate_dataset_integrity(dataset)

    # ── Debug mode adjustments ────────────────────────────────────────────────
    num_folds = 5
    if args.mode == 'debug':
        print(f"\n[DEBUG MODE] Subset of {len(dataset)} items active.")
        print(">> Restricting to 1 FOLD, 3 EPOCHS, and Batch 16 for rapid verification.")
        args.epoch = min(args.epoch, 3)
        args.batch = 16
        num_folds = 1 # Run only one fold in debug mode

    if args.task == 'classification':
        loss_fn = nn.BCEWithLogitsLoss()
        args.metrics = metrics_classification
    else:
        loss_fn = nn.MSELoss()
        args.metrics = metrics_regression

    # ── Reference Data Split & Cross-Validation Logic ─────────────────────────
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    
    save_path    = os.path.join(dataset_root, 'saved_models')
    # Include task in results_path so classification and regression outputs never overwrite each other
    results_path = os.path.join(dataset_root, 'results', data_name, args.task)
    os.makedirs(save_path,    exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    
    print(f"--- Path Diagnostics ---")
    print(f"Root Dir:            {dataset_root}")
    print(f"Results Path:        {os.path.abspath(results_path)}")
    print(f"Save Path:           {os.path.abspath(save_path)}")
    print(f"------------------------\n")

    best_results = {}
    # Two-level best-model tracking:
    # Level 1 — best epoch within each fold (handled by EarlyStopping in trainer)
    # Level 2 — best fold across all folds (tracked here)
    global_best_val_loss    = float('inf')
    global_best_model_state = None
    global_best_fold        = -1
  

    for fold in range(num_folds):
        print(f"\n===== FOLD {fold} =====")

        # Deterministic split: sort set before sampling so order is seed-stable
        if num_folds == 1:
            fold_size      = int(0.2 * len(indices))
            test_idx       = indices[:fold_size]
            train_pool_idx = sorted(set(indices) - set(test_idx))
            valid_idx      = random.Random(args.seed + fold).sample(
                                train_pool_idx, int(0.1 * len(train_pool_idx)))
            train_idx      = sorted(set(train_pool_idx) - set(valid_idx))
        else:
            fold_size = int(0.2 * len(indices))
            if fold == 4:
                test_idx = indices[fold * fold_size:]
            else:
                test_idx = indices[fold * fold_size:(fold + 1) * fold_size]

            train_pool_idx = sorted(set(indices) - set(test_idx))
            valid_idx      = random.Random(args.seed + fold).sample(
                                train_pool_idx, int(0.1 * len(train_pool_idx)))
            train_idx      = sorted(set(train_pool_idx) - set(valid_idx))

        # Reverting num_workers to 0. Since the dataset dynamically generates
        # ESM/ChemBERT embeddings using CUDA inside __getitem__, we cannot safely
        # use multiprocessing (which triggers the CUDA fork/spawn RuntimeError).
        num_workers = 0
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=args.batch, shuffle=True, collate_fn=moe_collate_fn,
            num_workers=num_workers, pin_memory=True
        )
        valid_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, valid_idx),
            batch_size=args.batch, shuffle=False, collate_fn=moe_collate_fn,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, test_idx),
            batch_size=args.batch, shuffle=False, collate_fn=moe_collate_fn,
            num_workers=num_workers, pin_memory=True
        )

        moe_model, optimizer = _build_experts_and_model(args, device, dataset)

        # Train (EarlyStopping inside picks best epoch → restores weights)
        moe_model, best_val_metrics, best_val_loss, fold_train_times, fold_val_times, epoch_history = train_moe(
            train_loader, moe_model, loss_fn, optimizer, args,
            valid_loader=valid_loader
        )

        # ── Save per-fold epoch history for graph plotting ─────────────────────
        fold_csv = os.path.join(results_path, f'fold_{fold}_epoch_history.csv')
        pd.DataFrame(epoch_history).to_csv(fold_csv, index=False)
        print(f"  Epoch history saved → {fold_csv}")

        # ── Level-2: update global best across folds ──────────────────────────────
        if best_val_loss < global_best_val_loss:
            global_best_val_loss    = best_val_loss
            # Strip DataParallel 'module.' prefix so the saved checkpoint
            # is portable and can be loaded on a single-GPU machine too.
            raw_model = moe_model.module if hasattr(moe_model, 'module') else moe_model
            global_best_model_state = copy.deepcopy(raw_model.state_dict())
            global_best_fold        = fold
            print(f"  ✓ New global best model  fold={fold}  val_loss={best_val_loss:.4f}")

        # ── Inductive Conformal Prediction (ICP) - Reference Logic ────────────
        print(f"\n--- ICP Calibration (Confidence Threshold={args.confidence}) ---")
        cal_scores = get_calibration_scores(moe_model, valid_loader, args.task)
        print(f"  Calibration items: {len(cal_scores)}")

        # Test at the end of each fold (model already holds best-epoch weights)
        print(f"\n--- Testing Fold {fold} ---")
        # Passing cal_scores to test_moe triggers the reference ICP logic
        result, perf, t_loss = test_moe(test_loader, moe_model, loss_fn, args, 
                                        split=f'Test_Fold_{fold}', cal_scores=cal_scores)
        
        # --- Varying ICP Confidence Evaluation (Reference Style) ---
        thresholds = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
        icp_results_summary = []
        
        print(f"\n{'Threshold':<10} {'Model':<10} {'Count':<8} {'Accuracy':<10} {'Selection%':<12}")
        print("-" * 55)

        if args.task == 'classification':
            for threshold in thresholds:
                icp_acc, sel_rate, sub_count = calculate_icp_selective_metrics(result, args, threshold)
                print(f"{threshold:<10.2f} {'MoE':<10} {sub_count:<8} {icp_acc:<10.3f} {sel_rate*100:<12.1f}%")
                icp_results_summary.append({
                    'threshold': threshold,
                    'accuracy': icp_acc,
                    'selection_rate': sel_rate,
                    'count': sub_count
                })
            
            # Use the args.confidence threshold for the "official" final metrics recorded in best_results
            icp_acc, sel_rate, sub_count = calculate_icp_selective_metrics(result, args, args.confidence)
            icp_names = ['ICP_Sub_Accuracy', 'ICP_Selection_Rate', 'ICP_Sub_Count']
            icp_vals  = [icp_acc, sel_rate, sub_count]
        else:
            # For Regression: Calculate coverage at different confidence levels
            # In regression, confidence = 1 - alpha. 
            # If threshold is 0.9, we want alpha = 0.1
            for threshold in thresholds:
                # Need to recalculate q for each alpha
                alpha_val = 1.0 - threshold
                q_val = np.quantile(cal_scores, threshold) # threshold is 1-alpha
                
                label = result[args.label]
                low   = result['pred'] - q_val
                high  = result['pred'] + q_val
                
                coverage  = ((label >= low) & (label <= high)).mean()
                avg_width = 2 * q_val
                
                print(f"{threshold:<10.2f} {'MoE':<10} {len(label):<8} {coverage:<10.3f} {'100.0%':<12} (Width: {avg_width:.3f})")
                icp_results_summary.append({
                    'threshold': threshold,
                    'coverage': coverage,
                    'avg_width': avg_width
                })

            # Use args.confidence for final metrics
            q_final = np.quantile(cal_scores, args.confidence)
            label = result[args.label]
            low   = result['icp_low']
            high  = result['icp_high']
            
            coverage  = ((label >= result['pred'] - q_final) & (label <= result['pred'] + q_final)).mean()
            avg_width = 2 * q_final
            
            icp_names = ['ICP_Coverage', 'ICP_Avg_Width']
            icp_vals  = [coverage, avg_width]
            
        final_metrics = perf.iloc[0].to_dict()
        for name, val in zip(icp_names, icp_vals):
            final_metrics[name] = val
            
        print(f"Fold {fold} Test Results (including ICP):", final_metrics)

        best_results[f"fold_{fold}"] = {
            "best_val_metrics": best_val_metrics,
            "best_val_loss":    best_val_loss,
            "test_metrics":     final_metrics,
            # FIX 2: Guard against empty time lists (e.g. early stop at epoch 0)
            "avg_train_time":   float(np.mean(fold_train_times)) if fold_train_times else 0.0,
            "avg_val_time":     float(np.mean(fold_val_times))   if fold_val_times   else 0.0,
        }

    # ── Save ONE final model: best epoch of best fold ─────────────────────────
    if global_best_model_state is not None:
        # Include task in filename so classification and regression models coexist
        global_model_path = os.path.join(save_path, f"best_model_{data_name}_{args.task}.pt")
        torch.save({
            'state_dict':    global_best_model_state,
            'best_fold':     global_best_fold,
            'best_val_loss': global_best_val_loss,
            'dataset':       data_name,
            'task':          args.task,
            'top_k':         args.top_k,
        }, global_model_path)
        print(f"\n✓ Global best model saved — fold={global_best_fold}  "
              f"val_loss={global_best_val_loss:.4f}")
        print(f"  Path: {global_model_path}")

    # ── Save cross-fold summary (all test metrics) for graph plotting ──────────
    summary_rows = []
    for fold_key, fold_data in best_results.items():
        row = {'fold': fold_key, 'best_val_loss': fold_data['best_val_loss']}
        row.update({f'test_{k}': v for k, v in fold_data['test_metrics'].items()})
        row.update({f'val_{k}':  v for k, v in (fold_data['best_val_metrics'] or {}).items()})
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # Compute cross-fold mean ± std for all numeric columns and append as summary rows
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(summary_rows) > 1 and numeric_cols:
        mean_row = {'fold': 'mean'}
        std_row  = {'fold': 'std'}
        mean_row.update({c: summary_df[c].mean() for c in numeric_cols})
        std_row.update( {c: summary_df[c].std()  for c in numeric_cols})
        summary_df = pd.concat([summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    summary_csv = os.path.join(results_path, 'fold_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Fold summary (with mean±std) saved → {summary_csv}")

    # Log results locally
    log_file = os.path.join(dataset_root, 'experiment_results.json')
    log_entry = {
        'timestamp': time.ctime(),
        'dataset': data_name,
        'task': args.task,
        'top_k': args.top_k,
        'fold_results': best_results,
        'total_time': time.time() - start_all
    }
    
    entries = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                # FIX 10: Only swallow malformed JSON, not OS/permission errors
                print(f"[Warning] Could not parse {log_file}; starting fresh log.")
                entries = []
    entries.append(log_entry)
    with open(log_file, 'w') as f:
        json.dump(entries, f, indent=4, cls=NpEncoder)

    total_duration = time.time() - start_all
    print(f"\nAll experiments done in {total_duration:.2f}s.")

if __name__ == '__main__':
    main()
