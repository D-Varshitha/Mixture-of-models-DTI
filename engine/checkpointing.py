"""
checkpointing.py
================
Safe checkpointing and resume-training utilities for the DTI-MoE framework.

Two checkpoint types
--------------------
1. latest_checkpoint.pt  — written every epoch; used to resume interrupted runs.
2. best_model.pt         — written only when validation improves; the final deliverable.

Checkpoint dict schema
----------------------
{
    'epoch':             int,          # last completed epoch (0-indexed)
    'fold':              int,
    'model_state':       OrderedDict,  # raw model weights (no DataParallel prefix)
    'optimizer_state':   dict,
    'scheduler_state':   dict | None,
    'best_val_loss':     float,
    'best_val_metrics':  dict | None,
    'es_counter':        int,          # EarlyStopping patience counter
    'es_best_score':     float | None, # EarlyStopping best score
    'rng_state':         dict,         # all RNG states for reproducibility
    'args_snapshot':     dict,         # selected args fields for provenance
    'dataset':           str,
    'task':              str,
    'top_k':             int,
}
"""

from __future__ import annotations

import json
import os
import random
import signal
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def checkpoint_dir_for(root: str, dataset_name: str, task: str) -> str:
    """Return   <root>/checkpoints/<dataset>/<task>/   (does NOT create it)."""
    return os.path.join(root, "checkpoints", dataset_name, task)


def checkpoint_paths_for(
    root: str, dataset_name: str, task: str, fold: int
) -> Tuple[str, str, str]:
    """Return (latest_path, best_path, state_json_path) for a fold."""
    d = checkpoint_dir_for(root, dataset_name, task)
    return (
        os.path.join(d, f"fold_{fold}_latest_checkpoint.pt"),
        os.path.join(d, f"fold_{fold}_best_model.pt"),
        os.path.join(d, f"fold_{fold}_training_state.json"),
    )


# ---------------------------------------------------------------------------
# RNG capture / restore
# ---------------------------------------------------------------------------

def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy":  np.random.get_state(),
        "torch":  torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.ByteTensor(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(
            [torch.ByteTensor(s) for s in state["cuda"]]
        )


# ---------------------------------------------------------------------------
# Core save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    *,
    checkpoint_path:  str,
    model:            torch.nn.Module,
    optimizer:        torch.optim.Optimizer,
    epoch:            int,
    fold:             int,
    best_val_loss:    float,
    best_val_metrics: Optional[Dict],
    es_counter:       int,
    es_best_score:    Optional[float],
    scheduler=None,
    dataset:          str = "",
    task:             str = "",
    top_k:            int = 2,
    args_snapshot:    Optional[Dict] = None,
    capture_rng:      bool = True,
) -> None:
    """Atomically write a checkpoint file (write-then-rename)."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    raw_model = model.module if hasattr(model, "module") else model
    ckpt = {
        "epoch":             epoch,
        "fold":              fold,
        "model_state":       raw_model.state_dict(),
        "optimizer_state":   optimizer.state_dict(),
        "scheduler_state":   scheduler.state_dict() if scheduler else None,
        "best_val_loss":     best_val_loss,
        "best_val_metrics":  best_val_metrics,
        "es_counter":        es_counter,
        "es_best_score":     es_best_score,
        "rng_state":         capture_rng_state() if capture_rng else None,
        "args_snapshot":     args_snapshot or {},
        "dataset":           dataset,
        "task":              task,
        "top_k":             top_k,
    }
    # Atomic write: save to a temp file then rename (safe against mid-write crashes)
    tmp_path = checkpoint_path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model:           torch.nn.Module,
    optimizer:       torch.optim.Optimizer,
    device:          torch.device,
    scheduler=None,
    restore_rng:     bool = False,
) -> Dict[str, Any]:
    """
    Load a checkpoint from *checkpoint_path* and restore all states.

    Caller contract
    ---------------
    The model MUST already be on *device* (and optionally wrapped with
    CustomDataParallel) BEFORE this function is called.  We do NOT call
    model.to(device) here because:
      1. The inner DTI_Sparse_MoE weights are loaded into model.module
         (or model directly), which is already on the correct device.
      2. Calling .to(device) again after DataParallel wrapping would
         silently trigger a redundant full-model device transfer on
         every resume.

    Returns the raw checkpoint dict so the caller can recover epoch, fold,
    best_val_loss, early-stopping counters, etc.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"[Checkpoint] No checkpoint found at: {checkpoint_path}"
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ── model weights ─────────────────────────────────────────────────────────
    # Unwrap DataParallel to access the raw inner model.
    # Checkpoints are ALWAYS saved without the 'module.' prefix (see save_checkpoint)
    # so this correctly handles both single-GPU and multi-GPU resume.
    raw_model = model.module if hasattr(model, "module") else model
    state = ckpt["model_state"]
    missing, unexpected = raw_model.load_state_dict(state, strict=True)
    if missing:
        raise RuntimeError(f"[Checkpoint] Missing keys in checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"[Checkpoint] Unexpected keys in checkpoint: {unexpected}")
    # NOTE: No model.to(device) here — model is already on device (see caller contract above).

    # ── optimizer ────────────────────────────────────────────────────────────
    optimizer.load_state_dict(ckpt["optimizer_state"])
    # Move optimizer tensors to the correct device
    for state_val in optimizer.state.values():
        for k, v in state_val.items():
            if isinstance(v, torch.Tensor):
                state_val[k] = v.to(device)

    # ── scheduler ────────────────────────────────────────────────────────────
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    # ── RNG ──────────────────────────────────────────────────────────────────
    if restore_rng and ckpt.get("rng_state"):
        restore_rng_state(ckpt["rng_state"])

    return ckpt


# ---------------------------------------------------------------------------
# Human-readable JSON state (for quick inspection without loading .pt)
# ---------------------------------------------------------------------------

def save_training_state_json(
    state_json_path: str,
    epoch:           int,
    fold:            int,
    best_val_loss:   float,
    best_val_metrics: Optional[Dict],
    es_counter:      int,
    dataset:         str,
    task:            str,
    top_k:           int,
    timestamp:       Optional[str] = None,
) -> None:
    """Write a small JSON alongside the checkpoint for easy inspection."""
    os.makedirs(os.path.dirname(state_json_path), exist_ok=True)
    state = {
        "epoch":             epoch,
        "fold":              fold,
        "best_val_loss":     best_val_loss if np.isfinite(best_val_loss) else None,
        "best_val_metrics":  best_val_metrics,
        "es_counter":        es_counter,
        "dataset":           dataset,
        "task":              task,
        "top_k":             top_k,
        "timestamp":         timestamp or time.ctime(),
    }
    with open(state_json_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=_json_default)


def _json_default(obj):
    """Handle numpy scalars in JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)