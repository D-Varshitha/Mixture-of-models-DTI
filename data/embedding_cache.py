"""
embedding_cache.py
==================
Offline Pretrained Embedding Cache for the DTI-MoE gating pipeline.

Responsibility
--------------
* Build a dictionary-based cache:
      protein_id  -> token-level ESM2 embedding  [L_p, H_p]  (CPU float32)
      drug_id     -> token-level ChemBERT embedding [L_d, H_d] (CPU float32)
* Persist the cache to disk with torch.save so it can be reused across runs.
* Validate cache integrity on load (shape, dtype, coverage of all IDs).

Cache file layout
-----------------
  <root>/embeddings/<dataset_name>/
      protein_embeddings.pt   – {protein_id: Tensor[L, H_p]}
      drug_embeddings.pt      – {drug_id:    Tensor[L, H_d]}
      cache_metadata.json     – provenance / model names / sizes

This module is intentionally free of dependencies on MoEDataset so it can
be imported and tested standalone.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def cache_dir_for(root: str, dataset_name: str) -> str:
    """Return the standard cache directory path (does NOT create it)."""
    return os.path.join(root, "embeddings", dataset_name)


def cache_paths_for(root: str, dataset_name: str) -> Tuple[str, str, str]:
    """Return (protein_path, drug_path, metadata_path) for a dataset."""
    d = cache_dir_for(root, dataset_name)
    return (
        os.path.join(d, "protein_embeddings.pt"),
        os.path.join(d, "drug_embeddings.pt"),
        os.path.join(d, "cache_metadata.json"),
    )


def cache_exists(root: str, dataset_name: str) -> bool:
    """Return True only when ALL three cache files are present and non-empty."""
    prot_path, drug_path, meta_path = cache_paths_for(root, dataset_name)
    for p in (prot_path, drug_path, meta_path):
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_embedding_cache(
    generator,           # PretrainedEmbeddingGenerator instance
    drug_id_to_smiles:  Dict[str, str],
    prot_id_to_seq:     Dict[str, str],
    root:               str,
    dataset_name:       str,
    rebuild:            bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Generate embeddings for all unique drugs and proteins; persist to disk.

    If the cache already exists and ``rebuild=False``, the existing files are
    loaded instead of re-generated.

    Parameters
    ----------
    generator       : PretrainedEmbeddingGenerator (lazy-loads HF models)
    drug_id_to_smiles : mapping  drug_id  -> SMILES string
    prot_id_to_seq    : mapping  protein_id -> amino-acid sequence
    root            : project root directory
    dataset_name    : e.g. "davis", "kiba"
    rebuild         : force regeneration even if cache files exist

    Returns
    -------
    (protein_cache, drug_cache)
        protein_cache : {protein_id: Tensor[L_p, H_p]}  on CPU, float32
        drug_cache    : {drug_id:    Tensor[L_d, H_d]}  on CPU, float32
    """
    prot_path, drug_path, meta_path = cache_paths_for(root, dataset_name)

    if cache_exists(root, dataset_name) and not rebuild:
        return load_embedding_cache(root, dataset_name,
                                    set(drug_id_to_smiles.keys()),
                                    set(prot_id_to_seq.keys()))

    # ── directories ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(prot_path), exist_ok=True)

    # ── generate drug embeddings ─────────────────────────────────────────────
    print(f"\n[EmbCache] Generating embeddings for {len(drug_id_to_smiles)} "
          f"unique drugs  ({dataset_name}) …")
    t0 = time.time()
    drug_cache: Dict[str, torch.Tensor] = {}
    for i, (did, smiles) in enumerate(drug_id_to_smiles.items()):
        emb = generator.embed_drug(smiles).to(torch.float32)   # CPU
        _validate_single_embedding(emb, f"drug[{did}]",
                                   expected_dim=generator.chembert_hidden_size)
        drug_cache[did] = emb
        if (i + 1) % 50 == 0 or (i + 1) == len(drug_id_to_smiles):
            print(f"  Drugs  : {i + 1:>5}/{len(drug_id_to_smiles)}")
    drug_gen_time = time.time() - t0
    print(f"  ✓ Drug embeddings done in {drug_gen_time:.1f}s")

    # ── generate protein embeddings ──────────────────────────────────────────
    print(f"\n[EmbCache] Generating embeddings for {len(prot_id_to_seq)} "
          f"unique proteins ({dataset_name}) …")
    t0 = time.time()
    prot_cache: Dict[str, torch.Tensor] = {}
    for i, (pid, seq) in enumerate(prot_id_to_seq.items()):
        emb = generator.embed_protein(seq).to(torch.float32)   # CPU
        _validate_single_embedding(emb, f"protein[{pid}]",
                                   expected_dim=generator.esm_hidden_size)
        prot_cache[pid] = emb
        if (i + 1) % 50 == 0 or (i + 1) == len(prot_id_to_seq):
            print(f"  Proteins: {i + 1:>5}/{len(prot_id_to_seq)}")
    prot_gen_time = time.time() - t0
    print(f"  ✓ Protein embeddings done in {prot_gen_time:.1f}s")

    # ── save to disk ─────────────────────────────────────────────────────────
    print(f"\n[EmbCache] Saving cache to {os.path.dirname(prot_path)} …")
    torch.save(prot_cache, prot_path)
    torch.save(drug_cache, drug_path)

    # Write metadata for provenance / debugging
    metadata = {
        "dataset":              dataset_name,
        "esm_model_name":       generator.esm_model_name,
        "chembert_model_name":  generator.chembert_model_name,
        "esm_hidden_size":      generator.esm_hidden_size,
        "chembert_hidden_size": generator.chembert_hidden_size,
        "num_unique_proteins":  len(prot_cache),
        "num_unique_drugs":     len(drug_cache),
        "protein_chunk_len":    generator.protein_chunk_len,
        "protein_chunk_stride": generator.protein_chunk_stride,
        "drug_chunk_len":       generator.drug_chunk_len,
        "drug_chunk_stride":    generator.drug_chunk_stride,
        "drug_gen_time_s":      round(drug_gen_time, 2),
        "prot_gen_time_s":      round(prot_gen_time, 2),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[EmbCache] Cache saved successfully.")
    print(f"  Proteins : {len(prot_cache)}")
    print(f"  Drugs    : {len(drug_cache)}")
    return prot_cache, drug_cache


# ---------------------------------------------------------------------------
# Cache loader
# ---------------------------------------------------------------------------

def load_embedding_cache(
    root:               str,
    dataset_name:       str,
    expected_drug_ids:  Optional[Iterable] = None,
    expected_prot_ids:  Optional[Iterable] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Load and validate the pre-existing embedding cache from disk.

    Parameters
    ----------
    root, dataset_name : same as in build_embedding_cache
    expected_drug_ids  : if provided, checks every ID is present in cache
    expected_prot_ids  : if provided, checks every ID is present in cache

    Returns
    -------
    (protein_cache, drug_cache)
    """
    prot_path, drug_path, meta_path = cache_paths_for(root, dataset_name)

    if not cache_exists(root, dataset_name):
        raise FileNotFoundError(
            f"[EmbCache] Cache not found for '{dataset_name}'. "
            f"Run with --rebuild-cache or mode='generate' first.\n"
            f"  Expected: {prot_path}"
        )

    t0 = time.time()
    # Load with weights_only=False because we saved plain dicts of tensors.
    prot_cache: Dict[str, torch.Tensor] = torch.load(
        prot_path, map_location="cpu", weights_only=False
    )
    drug_cache: Dict[str, torch.Tensor] = torch.load(
        drug_path, map_location="cpu", weights_only=False
    )
    load_time = time.time() - t0

    print(f"\n[EmbCache] Cache loaded from disk ({load_time:.2f}s)")
    print(f"  Loaded {len(prot_cache):>5} unique protein embeddings")
    print(f"  Loaded {len(drug_cache):>5} unique drug embeddings")

    # ── coverage validation ───────────────────────────────────────────────────
    if expected_drug_ids is not None:
        _check_coverage(drug_cache,  set(expected_drug_ids),  "drug")
    if expected_prot_ids is not None:
        _check_coverage(prot_cache,  set(expected_prot_ids),  "protein")

    # ── dtype + shape sanity (spot-check first entry of each) ────────────────
    _spot_check_cache(prot_cache, "protein")
    _spot_check_cache(drug_cache, "drug")

    print("[EmbCache] Embedding cache loaded successfully ✓")
    return prot_cache, drug_cache


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_single_embedding(
    emb: torch.Tensor,
    label: str,
    expected_dim: int,
) -> None:
    """Raise if an embedding tensor has wrong shape / dtype / NaN/Inf."""
    if emb.ndim != 2:
        raise RuntimeError(
            f"[EmbCache] {label}: expected 2-D tensor [L, H], got shape {tuple(emb.shape)}"
        )
    if emb.shape[0] == 0:
        raise RuntimeError(
            f"[EmbCache] {label}: zero-length sequence dimension (shape {tuple(emb.shape)})"
        )
    if emb.shape[1] != expected_dim:
        raise RuntimeError(
            f"[EmbCache] {label}: hidden dim mismatch — "
            f"expected {expected_dim}, got {emb.shape[1]}"
        )
    if torch.isnan(emb).any() or torch.isinf(emb).any():
        raise RuntimeError(
            f"[EmbCache] {label}: NaN or Inf found in embedding!"
        )
    if emb.device.type != "cpu":
        raise RuntimeError(
            f"[EmbCache] {label}: embedding must be on CPU before saving, "
            f"got device={emb.device}"
        )


def _check_coverage(
    cache: Dict[str, torch.Tensor],
    expected_ids: set,
    kind: str,
) -> None:
    missing = expected_ids - set(cache.keys())
    if missing:
        sample = sorted(missing)[:5]
        raise KeyError(
            f"[EmbCache] {len(missing)} {kind} ID(s) missing from cache. "
            f"First few: {sample}\n"
            f"Re-run with --rebuild-cache to regenerate."
        )


def _spot_check_cache(
    cache: Dict[str, torch.Tensor],
    kind: str,
) -> None:
    if not cache:
        raise RuntimeError(f"[EmbCache] {kind} cache is empty!")
    first_id, first_emb = next(iter(cache.items()))
    if not isinstance(first_emb, torch.Tensor):
        raise TypeError(
            f"[EmbCache] {kind} cache entry '{first_id}' is not a Tensor "
            f"(got {type(first_emb).__name__})"
        )
    if first_emb.ndim != 2 or first_emb.shape[0] == 0 or first_emb.shape[1] == 0:
        raise RuntimeError(
            f"[EmbCache] {kind} cache entry '{first_id}' has unexpected shape "
            f"{tuple(first_emb.shape)}"
        )
    if first_emb.dtype != torch.float32:
        raise RuntimeError(
            f"[EmbCache] {kind} cache entry '{first_id}' dtype is "
            f"{first_emb.dtype}, expected torch.float32"
        )
