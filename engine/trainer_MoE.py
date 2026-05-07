"""
trainer_MoE.py
==============
Training and evaluation loop for the DTI Sparse MoE model.

Changes over original
---------------------
* Checkpointing: saves latest_checkpoint.pt every epoch and best_model.pt
  whenever validation improves.
* Resume support: accepts start_epoch / restored early-stopping state so
  interrupted runs can continue exactly where they left off.
* Graceful interrupt: catches KeyboardInterrupt and CUDA OOM, saves an
  emergency checkpoint before re-raising.
* Corrected: optimizer tensors are moved to device after state_dict load.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from engine.checkpointing import (
    emergency_save,
    save_checkpoint,
    save_training_state_json,
    checkpoint_paths_for,
)
from engine.metrics import calculate_performance


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 15, verbose: bool = False, delta: float = 0.0):
        self.patience  = patience
        self.verbose   = verbose
        self.counter   = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta     = delta
        self.best_model_state: Optional[Dict] = None

    # ── state dict helpers (for checkpoint save/restore) ─────────────────────

    def state_dict(self) -> Dict[str, Any]:
        return {
            "counter":    self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "val_loss_min": self.val_loss_min,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.counter      = d["counter"]
        self.best_score   = d["best_score"]
        self.early_stop   = d["early_stop"]
        self.val_loss_min = d["val_loss_min"]

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        if self.verbose:
            print(f"  Validation loss improved "
                  f"({self.val_loss_min:.6f} → {val_loss:.6f}).")
        self.val_loss_min = val_loss
        # Strip DataParallel 'module.' wrapper for portability
        raw_model = model.module if hasattr(model, "module") else model
        self.best_model_state = copy.deepcopy(raw_model.state_dict())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_moe(
    train_loader:     torch.utils.data.DataLoader,
    model:            torch.nn.Module,
    loss_fn:          torch.nn.Module,
    optimizer:        torch.optim.Optimizer,
    args,
    valid_loader:     Optional[torch.utils.data.DataLoader] = None,
    # ── checkpoint / resume params ─────────────────────────────────────────
    checkpoint_dir:   Optional[str] = None,
    fold:             int = 0,
    dataset_name:     str = "",
    start_epoch:      int = 0,
    restored_es_state: Optional[Dict] = None,
    scheduler         = None,
) -> Tuple[
    torch.nn.Module,           # model (best weights restored)
    Optional[Dict],            # best_val_metrics
    float,                     # best_val_loss
    List[float],               # per-epoch train times
    List[float],               # per-epoch val times
    List[Dict],                # epoch_history rows
]:
    """
    Train the MoE model for one fold.

    Parameters
    ----------
    start_epoch       : first epoch to run (>0 when resuming)
    restored_es_state : EarlyStopping state dict loaded from a checkpoint
    checkpoint_dir    : folder to write latest/best .pt files; None disables

    Returns
    -------
    (model, best_val_metrics, best_val_loss, train_times, val_times,
     epoch_history)
    """
    # ── Early stopping ───────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=15, verbose=args.print_out)
    if restored_es_state is not None:
        early_stopping.load_state_dict(restored_es_state)
        print(f"  [Checkpoint] EarlyStopping restored — "
              f"counter={early_stopping.counter}, "
              f"best_val_loss={early_stopping.val_loss_min:.4f}")

    best_val_metrics: Optional[Dict] = None
    best_val_loss = early_stopping.val_loss_min  # preserves best seen so far

    train_times:   List[float] = []
    val_times:     List[float] = []
    epoch_history: List[Dict]  = []

    # Resolve checkpoint paths
    if checkpoint_dir is not None:
        latest_path, best_ckpt_path, state_json_path = checkpoint_paths_for(
            checkpoint_dir, dataset_name, args.task, fold
        )
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    else:
        latest_path = best_ckpt_path = state_json_path = None

    total_train_start = time.time()

    # ── Epoch loop ───────────────────────────────────────────────────────────
    interrupted = False
    for epoch in range(start_epoch, args.epoch):
        try:
            model.train()
            epoch_loss       = 0.0
            epoch_main_loss  = 0.0
            epoch_aux_loss   = 0.0
            ep_start         = time.time()
            num_batches      = len(train_loader)
            optimizer.zero_grad()

            for i, batch in enumerate(train_loader):
                output, aux_loss = model(batch)
                labels = batch["label"].to(output.device).float()

                main_loss = loss_fn(output, labels)

                # Guard: DataParallel returns per-GPU aux losses as a vector
                if aux_loss.dim() > 0:
                    aux_loss = aux_loss.mean()

                total_loss = main_loss + aux_loss

                # Gradient accumulation
                (total_loss / args.accumulation_steps).backward()
                if (i + 1) % args.accumulation_steps == 0 or (i + 1) == num_batches:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss      += total_loss.item()
                epoch_main_loss += main_loss.item()
                epoch_aux_loss  += aux_loss.item()

            train_time = time.time() - ep_start
            train_times.append(train_time)

            avg_main = epoch_main_loss / num_batches
            avg_aux  = epoch_aux_loss  / num_batches

            if args.print_out:
                print(f"\n--- Epoch {epoch} ---")
                print(f"  Train time : {train_time:.2f}s")
                print(f"  Main loss  : {avg_main:.4f}  |  Aux loss : {avg_aux:.4f}")

            # LR decay
            if args.lr_decay and args.decay_interval and epoch % args.decay_interval == 0:
                optimizer.param_groups[0]["lr"] *= args.lr_decay
            if scheduler is not None:
                scheduler.step()

            row: Dict = {
                "epoch":           epoch,
                "train_main_loss": avg_main,
                "train_aux_loss":  avg_aux,
            }

            # ── Validation ───────────────────────────────────────────────────
            v_loss = float("inf")
            if valid_loader:
                v_start = time.time()
                _, perf_df, v_loss = test_moe(
                    valid_loader, model, loss_fn, args, split="Valid"
                )
                val_time = time.time() - v_start
                val_times.append(val_time)

                val_metrics = perf_df.iloc[0].to_dict()

                if args.print_out:
                    print(f"  Val time   : {val_time:.2f}s")
                    print(f"  Val loss   : {v_loss:.4f}")
                    print(f"  Val metrics: {val_metrics}")

                if v_loss < best_val_loss:
                    best_val_loss    = v_loss
                    best_val_metrics = val_metrics

                row["val_loss"] = v_loss
                for mname, mval in val_metrics.items():
                    row[f"val_{mname}"] = mval

                early_stopping(v_loss, model)

                # ── Save best-model checkpoint ────────────────────────────────
                if (best_ckpt_path is not None
                        and early_stopping.counter == 0):
                    _save_both_checkpoints(
                        latest_path, best_ckpt_path, state_json_path,
                        model, optimizer, epoch, fold,
                        best_val_loss, best_val_metrics,
                        early_stopping, scheduler, args,
                        dataset_name, is_best=True,
                    )
                    print(f"  [Checkpoint] Best model saved  → epoch {epoch}")

            # ── Save latest checkpoint (every epoch) ─────────────────────────
            if latest_path is not None:
                _save_latest_checkpoint(
                    latest_path, state_json_path,
                    model, optimizer, epoch, fold,
                    best_val_loss, best_val_metrics,
                    early_stopping, scheduler, args, dataset_name,
                )
                print(f"  [Checkpoint] Latest checkpoint → epoch {epoch}")

            epoch_history.append(row)

            if valid_loader and early_stopping.early_stop:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

        except KeyboardInterrupt:
            print(f"\n[Trainer] KeyboardInterrupt received at epoch {epoch}.")
            interrupted = True
            if latest_path is not None:
                emergency_save(
                    checkpoint_path  = latest_path,
                    model            = model,
                    optimizer        = optimizer,
                    epoch            = epoch,
                    fold             = fold,
                    best_val_loss    = best_val_loss,
                    best_val_metrics = best_val_metrics,
                    es_counter       = early_stopping.counter,
                    es_best_score    = early_stopping.best_score,
                    scheduler        = scheduler,
                    dataset          = dataset_name,
                    task             = args.task,
                    top_k            = args.top_k,
                )
            raise  # re-raise so outer loop can catch / exit cleanly

        except RuntimeError as exc:
            oom = "out of memory" in str(exc).lower()
            tag  = "CUDA OOM" if oom else "RuntimeError"
            print(f"\n[Trainer] {tag} at epoch {epoch}: {exc}")
            if latest_path is not None:
                emergency_save(
                    checkpoint_path  = latest_path,
                    model            = model,
                    optimizer        = optimizer,
                    epoch            = epoch,
                    fold             = fold,
                    best_val_loss    = best_val_loss,
                    best_val_metrics = best_val_metrics,
                    es_counter       = early_stopping.counter,
                    es_best_score    = early_stopping.best_score,
                    dataset          = dataset_name,
                    task             = args.task,
                    top_k            = args.top_k,
                )
            raise

    total_train_time = time.time() - total_train_start
    print(f"\n  Total training time: {total_train_time:.2f}s")

    # Restore best weights found during this fold
    if early_stopping.best_model_state is not None:
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(early_stopping.best_model_state)

    return (
        model,
        best_val_metrics,
        best_val_loss,
        train_times,
        val_times,
        epoch_history,
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def test_moe(
    loader:       torch.utils.data.DataLoader,
    model:        torch.nn.Module,
    loss_fn:      torch.nn.Module,
    args,
    split:        str = "Test",
    cal_scores:   Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    from engine.conformal import apply_icp_reference_logic

    model.eval()
    preds, labels, com_ids, pro_ids = [], [], [], []
    icp_preds, icp_confs            = [], []
    icp_lows,  icp_highs            = [], []
    total_loss = total_main_loss = total_aux_loss = 0.0

    # Pre-compute regression quantile once (speed)
    q_val: Optional[float] = None
    if args.task == "regression" and cal_scores is not None and len(cal_scores) > 0:
        q_val = float(np.quantile(cal_scores, args.confidence))

    with torch.no_grad():
        for batch in loader:
            output, aux_loss = model(batch)
            batch_labels = batch["label"].to(output.device).float()
            main_loss    = loss_fn(output, batch_labels)

            if aux_loss.dim() > 0:
                aux_loss = aux_loss.mean()

            loss             = main_loss + aux_loss
            total_loss      += loss.item()
            total_main_loss += main_loss.item()
            total_aux_loss  += aux_loss.item()

            if args.task == "classification":
                probs = torch.sigmoid(output).cpu().numpy().tolist()
                preds.extend(probs)
                if cal_scores is not None:
                    icp_res = apply_icp_reference_logic(
                        output, cal_scores, args.task, alpha=1.0 - args.confidence
                    )
                    icp_preds.extend([r[0] for r in icp_res])
                    icp_confs.extend([r[1] for r in icp_res])
            else:
                preds.extend(output.cpu().numpy().tolist())
                if cal_scores is not None:
                    icp_res = apply_icp_reference_logic(
                        output, cal_scores, args.task,
                        alpha=1.0 - args.confidence, q=q_val,
                    )
                    icp_lows.extend(icp_res["lower"].tolist())
                    icp_highs.extend(icp_res["upper"].tolist())

            labels.extend(batch_labels.cpu().numpy().tolist())
            com_ids.extend(batch["com_id"])
            pro_ids.extend(batch["pro_id"])

    res_dict: Dict[str, Any] = {
        "com_id":   com_ids,
        "pro_id":   pro_ids,
        "pred":     preds,
        args.label: labels,
    }
    if cal_scores is not None:
        if args.task == "classification":
            res_dict["predByICP"] = icp_preds
            res_dict["confICP"]   = icp_confs
        else:
            res_dict["icp_low"]  = icp_lows
            res_dict["icp_high"] = icp_highs

    result_df = pd.DataFrame(res_dict)
    metrics   = calculate_performance(result_df, args)
    n_batches = max(len(loader), 1)
    avg_loss  = total_loss / n_batches

    if args.print_out:
        print(f"  [{split}] Main: {total_main_loss/n_batches:.4f} | "
              f"Aux: {total_aux_loss/n_batches:.4f}")

    return result_df, pd.DataFrame([metrics], columns=args.metrics), avg_loss


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_args_snapshot(args) -> Dict:
    """Capture key args fields for checkpoint provenance."""
    fields = [
        "data", "task", "top_k", "epoch", "batch", "lr",
        "weight_decay", "lambda_aux", "seed", "accumulation_steps",
        "esm_model_name", "chembert_model_name",
    ]
    return {f: getattr(args, f, None) for f in fields}


def _save_latest_checkpoint(
    latest_path, state_json_path,
    model, optimizer, epoch, fold,
    best_val_loss, best_val_metrics,
    early_stopping, scheduler, args, dataset_name,
) -> None:
    save_checkpoint(
        checkpoint_path  = latest_path,
        model            = model,
        optimizer        = optimizer,
        epoch            = epoch,
        fold             = fold,
        best_val_loss    = best_val_loss,
        best_val_metrics = best_val_metrics,
        es_counter       = early_stopping.counter,
        es_best_score    = early_stopping.best_score,
        scheduler        = scheduler,
        dataset          = dataset_name,
        task             = args.task,
        top_k            = args.top_k,
        args_snapshot    = _build_args_snapshot(args),
        capture_rng      = True,
    )
    if state_json_path:
        save_training_state_json(
            state_json_path  = state_json_path,
            epoch            = epoch,
            fold             = fold,
            best_val_loss    = best_val_loss,
            best_val_metrics = best_val_metrics,
            es_counter       = early_stopping.counter,
            dataset          = dataset_name,
            task             = args.task,
            top_k            = args.top_k,
        )


def _save_both_checkpoints(
    latest_path, best_ckpt_path, state_json_path,
    model, optimizer, epoch, fold,
    best_val_loss, best_val_metrics,
    early_stopping, scheduler, args, dataset_name,
    is_best: bool = False,
) -> None:
    """Save latest AND best checkpoints atomically."""
    _save_latest_checkpoint(
        latest_path, state_json_path,
        model, optimizer, epoch, fold,
        best_val_loss, best_val_metrics,
        early_stopping, scheduler, args, dataset_name,
    )
    if is_best:
        save_checkpoint(
            checkpoint_path  = best_ckpt_path,
            model            = model,
            optimizer        = optimizer,
            epoch            = epoch,
            fold             = fold,
            best_val_loss    = best_val_loss,
            best_val_metrics = best_val_metrics,
            es_counter       = early_stopping.counter,
            es_best_score    = early_stopping.best_score,
            scheduler        = scheduler,
            dataset          = dataset_name,
            task             = args.task,
            top_k            = args.top_k,
            args_snapshot    = _build_args_snapshot(args),
            capture_rng      = True,
        )
        print(f"  [Checkpoint] Best model updated ✓")
