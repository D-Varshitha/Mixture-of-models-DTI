"""
main_MoE.py
===========
Entry-point for the DTI Sparse Mixture-of-Experts training pipeline.

Key upgrades over previous version
-----------------------------------
* Offline embedding cache (ESM2 + ChemBERT) — no HF model runs per batch.
* Safe checkpointing every epoch (latest + best).
* Resume training via --resume-training / --checkpoint-path.
* Graceful KeyboardInterrupt handling with emergency checkpoint save.
* Bug fix: uses dataset.chembert_hidden_size / esm_hidden_size (from cache)
  instead of generator.{property} (which would reload HF models).
* Bug fix: stale print message updated to reflect offline-cache pipeline.
* Bug fix: regression ICP table header corrected (Coverage / Width).
* Root path accepted from CLI (--root) and propagated to ALL path derivations.
"""

import copy
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.parallel import DataParallel

# ---------------------------------------------------------------------------
# NpEncoder — safe JSON serialization of numpy scalars / arrays
# ---------------------------------------------------------------------------

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# CustomDataParallel — dict-aware scatter for multi-GPU training
# ---------------------------------------------------------------------------

class CustomDataParallel(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        batch     = inputs[0]
        B         = batch["label"].shape[0]
        num_gpus  = len(device_ids)
        chunk_size = (B + num_gpus - 1) // num_gpus

        scattered_batches = []
        for i in range(num_gpus):
            start = i * chunk_size
            end   = min(start + chunk_size, B)
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

        return scattered_batches, [{}] * len(scattered_batches)


# ---------------------------------------------------------------------------
# Imports (after class definitions to avoid circular issues at module level)
# ---------------------------------------------------------------------------

from data.moe_dataset import MoEDataset, moe_collate_fn
from engine.checkpointing import (
    checkpoint_dir_for,
    checkpoint_paths_for,
    load_checkpoint,
)
from engine.conformal import apply_icp_reference_logic, get_calibration_scores
from engine.metrics import calculate_icp_selective_metrics
from engine.trainer_MoE import train_moe, test_moe
from models import build_model
from models.model_MoE import DTI_Sparse_MoE
from config import args, metrics_classification, metrics_regression


# ---------------------------------------------------------------------------
# Dataset integrity check
# ---------------------------------------------------------------------------

def _validate_dataset_integrity(dataset):
    """Verify all required sample keys and tensor shapes are present."""
    required_keys = [
        "label", "shared_drug", "shared_prot",
        "shared_drug_mask", "shared_prot_mask",
        "dpdta_com", "dpdta_pro",
        "dcdti_com", "dcdti_pro",
        "mdprd_com", "mdprd_pro",
        "gifdti_com", "gifdti_pro", "gifdti_com_mask", "gifdti_pro_mask",
        "pcpi_graph", "pcpi_morgan", "pcpi_sequence",
        "dp_af", "dp_bf", "dp_ag", "dp_bg", "dp_abn", "dp_pro",
    ]
    base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    sample_ids = sorted({0, max(0, len(base) // 2), max(0, len(base) - 1)})

    for sid in sample_ids:
        sample = base[sid]

        missing = [k for k in required_keys if k not in sample]
        if missing:
            raise RuntimeError(
                f"[Dataset Integrity] Missing keys in sample {sid}: {missing}"
            )
        empty = [
            k for k, v in sample.items()
            if (isinstance(v, torch.Tensor) and v.numel() == 0)
            or (isinstance(v, list) and len(v) == 0)
        ]
        if empty:
            raise RuntimeError(
                f"[Dataset Integrity] Empty tensors in sample {sid}: {empty}"
            )

        checks = {
            "dcdti_com  shape[-1] == 2048": sample["dcdti_com"].shape[-1] == 2048,
            "mdprd_com  shape[-1] == 1024": sample["mdprd_com"].shape[-1] == 1024,
            "pcpi_morgan shape[-1] == 1024": sample["pcpi_morgan"].shape[-1] == 1024,
            "mdprd_pro shape == (5,500,500)": tuple(sample["mdprd_pro"].shape) == (5, 500, 500),
            "shared_drug ndim == 2": sample["shared_drug"].ndim == 2,
            "shared_prot ndim == 2": sample["shared_prot"].ndim == 2,
            "dcdti_pro dtype == long": sample["dcdti_pro"].dtype == torch.long,
        }
        for desc, ok in checks.items():
            if not ok:
                raise RuntimeError(
                    f"[Dataset Integrity] {desc} failed in sample {sid}. "
                    f"Got: {sample.get(desc.split()[0], 'N/A')}"
                )

    print(f"[Dataset Integrity] All {len(required_keys)} required keys present. ✓")


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_experts_and_model(args, device, dataset):
    """Build fresh expert models + MoE wrapper + Adam optimizer."""
    experts_dict = {
        "dpdta":        build_model("dpdta",        args.task, com_len=args.com_len, pro_len=args.pro_len),
        "dcdti":        build_model("dcdti",        args.task, com_len=args.com_len, pro_len=args.pro_len),
        "dp":           build_model("dp",           args.task, com_len=args.com_len, pro_len=args.pro_len),
        "mdprd":        build_model("mdprd",        args.task, com_len=args.com_len, pro_len=args.pro_len),
        "gifdti":       build_model("gifdti",       args.task, com_len=args.com_len, pro_len=args.pro_len),
        "perceivercpi": build_model("perceivercpi", args.task, com_len=args.com_len, pro_len=args.pro_len),
    }

    # BUG FIX: use dataset.{hidden_size} (read from cache metadata) instead of
    # dataset.generator.{property} which triggers a full HF model forward pass.
    drug_dim = dataset.chembert_hidden_size
    prot_dim = dataset.esm_hidden_size

    moe_model = DTI_Sparse_MoE(
        experts_dict,
        drug_vocab          = 65,
        prot_vocab          = 26,
        k                   = args.top_k,
        lambda_aux          = args.lambda_aux,
        drug_pretrained_dim = drug_dim,
        prot_pretrained_dim = prot_dim,
    ).to(device)

    if torch.cuda.device_count() > 1 and args.device != "cpu":
        print(f"  Using {torch.cuda.device_count()} GPUs (CustomDataParallel).")
        moe_model = CustomDataParallel(moe_model)

    optimizer = torch.optim.Adam(
        moe_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    return moe_model, optimizer


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _fold_results_path(results_path: str, fold: int) -> str:
    """Path to the per-fold results JSON written after a fold fully completes."""
    return os.path.join(results_path, f"fold_{fold}_results.json")


def _save_fold_results(results_path: str, fold: int, fold_data: dict) -> None:
    """Persist fold results to disk so a resumed run can skip completed folds."""
    path = _fold_results_path(results_path, fold)
    try:
        with open(path, "w") as f:
            json.dump(fold_data, f, indent=2, cls=NpEncoder)
    except Exception as exc:
        print(f"  [Warning] Could not save fold {fold} results: {exc}")


def _load_fold_results(results_path: str, fold: int) -> dict | None:
    """Return saved fold results dict if the fold has already completed, else None."""
    path = _fold_results_path(results_path, fold)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"  [Warning] Could not load fold {fold} results from {path}: {exc}")
        return None


def _try_resume(args, moe_model, optimizer, device, dataset_root, data_name, fold):
    """
    If --resume-training is set, load the latest checkpoint for this fold.
    Returns (start_epoch, restored_es_state, prior_best_val_metrics)
    or (0, None, None) if not resuming.
    """
    if not getattr(args, "resume_training", False):
        return 0, None, None

    latest_path, _, _ = checkpoint_paths_for(dataset_root, data_name, args.task, fold)

    # Allow explicit override via --checkpoint-path
    explicit = getattr(args, "checkpoint_path", None)
    if explicit:
        latest_path = explicit

    if not os.path.isfile(latest_path):
        print(f"  [Resume] No checkpoint found at {latest_path}; starting fresh.")
        return 0, None, None

    ckpt = load_checkpoint(latest_path, moe_model, optimizer, device,
                           restore_rng=True)
    start_epoch = ckpt["epoch"] + 1
    es_state = {
        "counter":      ckpt.get("es_counter", 0),
        "best_score":   ckpt.get("es_best_score", None),
        "early_stop":   False,
        "val_loss_min": ckpt.get("best_val_loss", float("inf")),
    }
    prior_best_val_metrics = ckpt.get("best_val_metrics", None)
    best_loss = es_state["val_loss_min"]
    print(f"  [Checkpoint] Loaded checkpoint successfully")
    print(f"  [Resume] Resuming from epoch {start_epoch} (fold {ckpt.get('fold', fold)})")
    if best_loss != float("inf"):
        print(f"  [Resume] Best validation loss restored: {best_loss:.4f}")
    return start_epoch, es_state, prior_best_val_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_all = time.time()

    # ── Hardware ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device("cuda")
        print("Hardware detected: GPU")
    else:
        device = torch.device("cpu")
        print("Hardware detected: CPU")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_name    = args.data if args.data else "davis"
    label_col    = "lab" if args.task == "classification" else "affinity"
    args.label   = label_col
    # Root path: from --root CLI flag or cwd (consistent across ALL path derivations)
    dataset_root = args.root if args.root else os.getcwd()

    print(f"\nLoading dataset : {data_name}  |  task: {args.task}")
    print(f"  Root          : {dataset_root}")
    print(f"  Embeddings    : ESM2 + ChemBERT — offline cache (generated once, reused)")
    print(f"  Expert feats  : fingerprints / MPNN / MDeePred — disk-cached")
    print(f"  Checkpoints   : {os.path.join(dataset_root, 'checkpoints', data_name, args.task)}\n")

    current_subset = args.subset_size if args.mode == "debug" else None

    dataset = MoEDataset(
        root          = dataset_root,
        dataset_name  = data_name,
        label_type    = label_col,
        subset_size   = current_subset,
        MAX_SMI_LEN   = args.com_len,
        MAX_SEQ_LEN   = args.pro_len,
        mode          = args.get_dataset,
        rebuild_cache = args.rebuild_cache,
    )

    _validate_dataset_integrity(dataset)
    print(f"\n[Dataset] Total loaded samples: {len(dataset)}")

    # ── Debug mode ────────────────────────────────────────────────────────────
    num_folds = 5
    if args.mode == "debug":
        print(f"\n[DEBUG MODE] Subset of {len(dataset)} items active.")
        print(">> 1 FOLD | 3 EPOCHS | batch=16")
        args.epoch = min(args.epoch, 3)
        args.batch = 16
        num_folds  = 1

    if args.task == "classification":
        loss_fn     = nn.BCEWithLogitsLoss()
        args.metrics = metrics_classification
    else:
        loss_fn     = nn.MSELoss()
        args.metrics = metrics_regression

    # ── Deterministic splits ──────────────────────────────────────────────────
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    # All paths derived from dataset_root (accepts --root from CLI)
    results_path  = os.path.join(dataset_root, "results",     data_name, args.task)
    save_path     = os.path.join(dataset_root, "saved_models")
    ckpt_dir      = checkpoint_dir_for(dataset_root, data_name, args.task)

    for d in (results_path, save_path, ckpt_dir):
        os.makedirs(d, exist_ok=True)

    print("--- Path Diagnostics ---")
    print(f"  Root          : {dataset_root}")
    print(f"  Results       : {os.path.abspath(results_path)}")
    print(f"  Saved models  : {os.path.abspath(save_path)}")
    print(f"  Checkpoints   : {os.path.abspath(ckpt_dir)}")
    print("------------------------\n")

    best_results            = {}
    global_best_val_loss    = float("inf")
    global_best_model_state = None
    global_best_fold        = -1

    # ── Fold loop ─────────────────────────────────────────────────────────────
    for fold in range(num_folds):
        print(f"\n===== FOLD {fold} =====")

        # ── Resume: skip folds that already fully completed ───────────────────
        if getattr(args, "resume_training", False):
            saved = _load_fold_results(results_path, fold)
            if saved is not None:
                print(f"  [Resume] Fold {fold} already completed — restoring results from disk.")
                best_results[f"fold_{fold}"] = saved
                # Update global best from the restored results
                if saved["best_val_loss"] < global_best_val_loss:
                    global_best_val_loss = saved["best_val_loss"]
                    global_best_fold     = fold
                    # Global best model state: load from the saved best checkpoint
                    _, best_ckpt_path, _ = checkpoint_paths_for(
                        dataset_root, data_name, args.task, fold
                    )
                    if os.path.isfile(best_ckpt_path):
                        tmp_model, tmp_opt = _build_experts_and_model(args, device, dataset)
                        tmp_ckpt = load_checkpoint(best_ckpt_path, tmp_model, tmp_opt, device)
                        raw_model = tmp_model.module if hasattr(tmp_model, "module") else tmp_model
                        global_best_model_state = copy.deepcopy(raw_model.state_dict())
                        print(f"  ✓ Global best restored — fold={fold}  "
                              f"val_loss={global_best_val_loss:.4f}")
                    del tmp_model, tmp_opt
                continue  # skip to next fold

        # Deterministic 5-fold split
        fold_size = int(0.2 * len(indices))
        if num_folds == 1:
            test_idx = indices[:fold_size]
        elif fold == num_folds - 1:
            test_idx = indices[fold * fold_size:]
        else:
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]

        train_pool_idx = sorted(set(indices) - set(test_idx))
        valid_idx      = random.Random(args.seed + fold).sample(
            train_pool_idx, int(0.1 * len(train_pool_idx))
        )
        train_idx = sorted(set(train_pool_idx) - set(valid_idx))

        print(f"  Split sizes -> Train: {len(train_idx)} | Valid: {len(valid_idx)} | Test: {len(test_idx)}")

        num_workers  = 0   # keep 0: dataset uses in-process cache lookups
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=args.batch, shuffle=True,
            collate_fn=moe_collate_fn, num_workers=num_workers, pin_memory=False,
        )
        valid_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, valid_idx),
            batch_size=args.batch, shuffle=False,
            collate_fn=moe_collate_fn, num_workers=num_workers, pin_memory=False,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, test_idx),
            batch_size=args.batch, shuffle=False,
            collate_fn=moe_collate_fn, num_workers=num_workers, pin_memory=False,
        )

        moe_model, optimizer = _build_experts_and_model(args, device, dataset)

        # ── Resume ───────────────────────────────────────────────────────────
        start_epoch, restored_es_state, prior_best_val_metrics = _try_resume(
            args, moe_model, optimizer, device, dataset_root, data_name, fold
        )

        # ── Load prior epoch history (preserve pre-interrupt rows on resume) ──
        fold_csv      = os.path.join(results_path, f"fold_{fold}_epoch_history.csv")
        prior_history = []
        if start_epoch > 0 and os.path.isfile(fold_csv):
            try:
                prior_history = pd.read_csv(fold_csv).to_dict("records")
                print(f"  [Resume] Loaded {len(prior_history)} prior epoch(s) from history CSV.")
            except Exception as exc:
                print(f"  [Warning] Could not load prior epoch history: {exc}")

        # ── Train ────────────────────────────────────────────────────────────
        try:
            moe_model, best_val_metrics, best_val_loss, fold_train_times, \
                fold_val_times, epoch_history = train_moe(
                    train_loader, moe_model, loss_fn, optimizer, args,
                    valid_loader            = valid_loader,
                    checkpoint_dir          = dataset_root,
                    fold                    = fold,
                    dataset_name            = data_name,
                    start_epoch             = start_epoch,
                    restored_es_state       = restored_es_state,
                    prior_best_val_metrics  = prior_best_val_metrics,
                )
        except KeyboardInterrupt:
            print("\n[Main] Keyboard interrupt — stopping cross-validation.")
            sys.exit(0)

        # ── Save epoch history CSV (append prior rows on resume) ──────────────
        combined_history = prior_history + epoch_history
        pd.DataFrame(combined_history).to_csv(fold_csv, index=False)
        print(f"  Epoch history → {fold_csv}")

        # ── Global best across folds ──────────────────────────────────────────
        if best_val_loss < global_best_val_loss:
            global_best_val_loss    = best_val_loss
            raw_model               = moe_model.module if hasattr(moe_model, "module") else moe_model
            global_best_model_state = copy.deepcopy(raw_model.state_dict())
            global_best_fold        = fold
            print(f"  ✓ New global best — fold={fold}  val_loss={best_val_loss:.4f}")

        # ── ICP Calibration ───────────────────────────────────────────────────
        print(f"\n--- ICP Calibration--")
        cal_scores = get_calibration_scores(moe_model, valid_loader, args.task)
        print(f"  Calibration items: {len(cal_scores)}")

        # ── Test ──────────────────────────────────────────────────────────────
        print(f"\n--- Testing Fold {fold} ---")
        result, perf, t_loss = test_moe(
            test_loader, moe_model, loss_fn, args,
            split=f"Test_Fold_{fold}", cal_scores=cal_scores,
        )

        # ── ICP threshold sweep ───────────────────────────────────────────────
        thresholds       = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
        icp_results_summary = []
        icp_names = []
        icp_vals  = []

        if args.task == "classification":
            print(f"\n{'Threshold':<10} {'Model':<10} {'Count':<8} "
                  f"{'Accuracy':<10} {'Selection%':<12}")
            print("-" * 55)
            for thr in thresholds:
                icp_acc, sel_rate, sub_count = calculate_icp_selective_metrics(
                    result, args, thr
                )
                print(f"{thr:<10.2f} {'MoE':<10} {sub_count:<8} "
                      f"{icp_acc:<10.3f} {sel_rate*100:<12.1f}%")
                icp_results_summary.append({
                    "threshold":      thr,
                    "accuracy":       icp_acc,
                    "selection_rate": sel_rate,
                    "count":          sub_count,
                })
                # Add to final metrics dict for this specific threshold
                icp_names.extend([f"ICP_Acc_{thr}", f"ICP_Sel_{thr}", f"ICP_Count_{thr}"])
                icp_vals.extend([icp_acc, sel_rate, sub_count])

        else:
            print(f"\n{'Threshold':<10} {'Model':<10} {'Count':<8} "
                  f"{'Coverage':<10} {'Width':<12}")
            print("-" * 55)
            for thr in thresholds:
                q_val    = float(np.quantile(cal_scores, thr))
                lab_arr  = np.array(result[args.label])
                pred_arr = np.array(result["pred"])
                coverage = float(((lab_arr >= pred_arr - q_val) &
                                  (lab_arr <= pred_arr + q_val)).mean())
                width    = 2.0 * q_val
                print(f"{thr:<10.2f} {'MoE':<10} {len(lab_arr):<8} "
                      f"{coverage:<10.3f} {width:<12.3f}")
                icp_results_summary.append({
                    "threshold": thr,
                    "coverage":  coverage,
                    "avg_width": width,
                })
                # Add to final metrics dict for this specific threshold
                icp_names.extend([f"ICP_Cov_{thr}", f"ICP_Width_{thr}"])
                icp_vals.extend([coverage, width])

        # ── Save ICP CSV ──────────────────────────────────────────────────────
        if args.save_result:
            icp_df   = pd.DataFrame(icp_results_summary)
            icp_file = os.path.join(results_path, f"ICP_Threshold_Summary_Fold_{fold}.csv")
            icp_df.to_csv(icp_file, index=False)
            print(f"  ✓ ICP summary → {icp_file}")

        final_metrics = perf.iloc[0].to_dict()
        for name, val in zip(icp_names, icp_vals):
            final_metrics[name] = val
        print(f"Fold {fold} test results: {final_metrics}")

        fold_data = {
            "best_val_metrics": best_val_metrics,
            "best_val_loss":    best_val_loss,
            "test_metrics":     final_metrics,
            "avg_train_time":   float(np.mean(fold_train_times)) if fold_train_times else 0.0,
            "avg_val_time":     float(np.mean(fold_val_times))   if fold_val_times   else 0.0,
            "valid_samples":    len(valid_idx),
            "test_samples":     len(test_idx),
        }
        best_results[f"fold_{fold}"] = fold_data

        # ── Persist fold results so a future resume can skip this fold ────────
        _save_fold_results(results_path, fold, fold_data)

    # ── Save global best model ────────────────────────────────────────────────
    if global_best_model_state is not None:
        global_model_path = os.path.join(
            save_path, f"best_model_{data_name}_{args.task}.pt"
        )
        torch.save({
            "state_dict":    global_best_model_state,
            "best_fold":     global_best_fold,
            "best_val_loss": global_best_val_loss,
            "dataset":       data_name,
            "task":          args.task,
            "top_k":         args.top_k,
        }, global_model_path)
        print(f"\n✓ Global best model saved — fold={global_best_fold}  "
              f"val_loss={global_best_val_loss:.4f}")
        print(f"  Path: {global_model_path}")

    # ── Cross-fold summary CSV ────────────────────────────────────────────────
    summary_rows = []
    for fold_key, fold_data in best_results.items():
        row = {"fold": fold_key, "best_val_loss": fold_data["best_val_loss"]}
        if "valid_samples" in fold_data:
            row["valid_samples"] = fold_data["valid_samples"]
            row["test_samples"]  = fold_data["test_samples"]
        row.update({f"test_{k}": v for k, v in fold_data["test_metrics"].items()})
        row.update({f"val_{k}":  v for k, v in (fold_data["best_val_metrics"] or {}).items()})
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(summary_rows) > 1 and numeric_cols:
        mean_row = {"fold": "mean", **{c: summary_df[c].mean() for c in numeric_cols}}
        std_row  = {"fold": "std",  **{c: summary_df[c].std()  for c in numeric_cols}}
        summary_df = pd.concat(
            [summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True
        )
    summary_csv = os.path.join(results_path, "fold_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Fold summary → {summary_csv}")

    # ── Append to experiment log ──────────────────────────────────────────────
    log_file = os.path.join(dataset_root, "experiment_results.json")
    log_entry = {
        "timestamp":    time.ctime(),
        "dataset":      data_name,
        "task":         args.task,
        "top_k":        args.top_k,
        "fold_results": best_results,
        "total_time":   time.time() - start_all,
    }
    entries = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                print(f"[Warning] Could not parse {log_file}; starting fresh log.")
    entries.append(log_entry)
    with open(log_file, "w") as f:
        json.dump(entries, f, indent=4, cls=NpEncoder)

    print(f"\nAll experiments done in {time.time() - start_all:.2f}s.")


if __name__ == "__main__":
    main()
