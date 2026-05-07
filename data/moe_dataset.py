"""
moe_dataset.py
==============
MoEDataset — PyTorch Dataset that assembles ALL expert inputs plus the
shared gating embeddings for the DTI Sparse MoE model.

Embedding strategy (offline cache)
-----------------------------------
ESM2 (protein) and ChemBERT (drug) token-level embeddings are generated
ONCE per unique molecule / protein and stored to disk:

    <root>/embeddings/<dataset_name>/
        protein_embeddings.pt  – {protein_id: Tensor[L_p, H_p]}
        drug_embeddings.pt     – {drug_id:    Tensor[L_d, H_d]}

During __getitem__ the correct embedding is fetched from the in-memory
cache dict using the sample's drug/protein ID — NO forward pass through
any HF model occurs during training / evaluation.

If the cache files do not exist they are generated and saved automatically
before the first epoch starts.  Pass ``rebuild_cache=True`` to force
regeneration (useful after swapping the underlying HF model).

The expert-specific features (fingerprints, MPNN graphs, MDeePred
matrices, …) are unchanged from the original implementation.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from config import args
from .dataset import CPIDataset, CHARPROTSET, CHARCANSMISET, CHARISOSMISET
from .embedding_cache import (
    build_embedding_cache,
    load_embedding_cache,
    cache_exists,
    _validate_single_embedding,
)
from .pretrained_embeddings import PretrainedEmbeddingGenerator
from .utils import build_seq_enc, build_ngram, build_one_hot_enc

_REQUIRED_COM_FILES = {
    'fp_2048': 'fps_2048.npy',
    'fp_1024': 'fps_1024.npy',
    'dp_mpnn': 'dp_mpnn.npy',
}
_REQUIRED_PRO_FILES = {
    'mdprd_pro': 'mdprd_pro.pth',
    'dp_pro':    'dp_pro.npy',
}


class MoEDataset(CPIDataset):
    """
    DTI dataset that serves every input needed by the Sparse MoE model.

    Parameters
    ----------
    root            : project root (contains dataset/ and embeddings/ dirs)
    dataset_name    : e.g. "davis", "kiba"
    MAX_SMI_LEN     : max SMILES length for expert character encodings
    MAX_SEQ_LEN     : max sequence length for expert amino-acid encodings
    label_type      : column name for labels ("label" / "affinity")
    mode            : "generate" → regenerate expert feature files if missing
                      "load"     → assume expert files already exist
    subset_size     : optional int for debug sub-sampling
    rebuild_cache   : if True, force regeneration of the embedding cache even
                      when cache files already exist on disk
    """

    def __init__(
        self,
        root:           str,
        dataset_name:   str,
        MAX_SMI_LEN:    int  = 100,
        MAX_SEQ_LEN:    int  = 1000,
        label_type:     str  = 'label',
        mode:           str  = 'generate',
        subset_size:    Optional[int] = None,
        rebuild_cache:  bool = False,
    ):
        super(MoEDataset, self).__init__(
            root, dataset_name, label_type, mode, subset_size=subset_size
        )

        self.MAX_SMI_LEN   = MAX_SMI_LEN
        self.MAX_SEQ_LEN   = MAX_SEQ_LEN
        self.mode          = mode
        self.rebuild_cache = rebuild_cache

        # ── 1. Build / load the pretrained embedding generator ──────────────
        # The generator wraps ESM2 + ChemBERT; HF models are loaded lazily
        # only when they are actually needed (i.e., cache is missing).
        self.generator = PretrainedEmbeddingGenerator(
            esm_model_name       = args.esm_model_name,
            chembert_model_name  = args.chembert_model_name,
            cache_dir            = args.hf_cache_dir,
            device               = 'cuda' if (torch.cuda.is_available()
                                              and args.device != 'cpu') else 'cpu',
            protein_chunk_len    = args.protein_chunk_len,
            protein_chunk_stride = args.protein_chunk_stride,
            drug_chunk_len       = args.drug_chunk_len,
            drug_chunk_stride    = args.drug_chunk_stride,
        )

        # ── 2. Offline embedding cache ───────────────────────────────────────
        # Build {id: SMILES} / {id: seq} mappings over the UNIQUE molecules
        # so we never embed the same molecule twice.
        drug_id_to_smiles: Dict[str, str] = dict(self.lig_dic)   # {lig_id: smiles}
        prot_id_to_seq:    Dict[str, str] = dict(self.pro_dic)    # {pro_id: seq}

        print(f"\n[MoEDataset] Unique drugs    : {len(drug_id_to_smiles)}")
        print(f"[MoEDataset] Unique proteins : {len(prot_id_to_seq)}")

        # build_embedding_cache handles the generate-or-load decision
        self._prot_emb_cache, self._drug_emb_cache = build_embedding_cache(
            generator         = self.generator,
            drug_id_to_smiles = drug_id_to_smiles,
            prot_id_to_seq    = prot_id_to_seq,
            root              = root,
            dataset_name      = dataset_name,
            rebuild           = rebuild_cache,
        )

        # Expose hidden sizes (read from cache metadata without loading HF models
        # if the cache already existed)
        self.chembert_hidden_size = next(iter(self._drug_emb_cache.values())).shape[1]
        self.esm_hidden_size      = next(iter(self._prot_emb_cache.values())).shape[1]

        print(f"[MoEDataset] Drug  embedding dim : {self.chembert_hidden_size}")
        print(f"[MoEDataset] Protein embedding dim: {self.esm_hidden_size}")

        # ── 3. Expert feature files ──────────────────────────────────────────
        if mode == 'generate':
            print(f"\n[MoEDataset] Generating expert features for '{dataset_name}' …")
            os.makedirs(self.com_feat_dir, exist_ok=True)
            os.makedirs(self.pro_feat_dir, exist_ok=True)
            self.process_molecule(MAX_SMI_LEN, 'fp_2048')
            self.process_molecule(MAX_SMI_LEN, 'fp_1024')
            self.process_molecule(MAX_SMI_LEN, 'mpnn')
            self.process_protein(MAX_SEQ_LEN,  'mdprd')
            self.process_protein(MAX_SEQ_LEN,  'dp')

        self._validate_required_feature_files()
        self.fp_2048     = self.get_fp(2048)
        self.fp_1024     = self.get_fp(1024)
        self.mpnn_data   = self.get_mpnn_feature
        self.dp_pro_data = self.get_dp
        self.mdprd_data  = self.get_mdprd

        self.validate_features()

    # ── public interface ─────────────────────────────────────────────────────

    def __len__(self):
        return self.num_interaction

    # ── cache retrieval ──────────────────────────────────────────────────────

    def _get_drug_embedding(self, drug_id: str) -> torch.Tensor:
        """Fetch the cached ChemBERT token embedding for *drug_id*.

        Raises a descriptive ``KeyError`` if the ID is missing from the cache
        (which should never happen after a successful cache build).
        """
        try:
            emb = self._drug_emb_cache[drug_id]
        except KeyError:
            raise KeyError(
                f"[MoEDataset] Drug ID '{drug_id}' not found in embedding cache. "
                f"Cache contains {len(self._drug_emb_cache)} entries. "
                f"Re-run with --rebuild-cache."
            )
        return emb  # CPU float32, shape [L_d, H_d]

    def _get_prot_embedding(self, prot_id: str) -> torch.Tensor:
        """Fetch the cached ESM2 token embedding for *prot_id*."""
        try:
            emb = self._prot_emb_cache[prot_id]
        except KeyError:
            raise KeyError(
                f"[MoEDataset] Protein ID '{prot_id}' not found in embedding cache. "
                f"Cache contains {len(self._prot_emb_cache)} entries. "
                f"Re-run with --rebuild-cache."
            )
        return emb  # CPU float32, shape [L_p, H_p]

    # ── main data retrieval ──────────────────────────────────────────────────

    def __getitem__(self, ind):
        cid       = self.lig_mapping.inverse[self.lig[ind]]   # integer index → str ID
        pid       = self.pro_mapping.inverse[self.pro[ind]]
        raw_label = self.label[ind]

        # Convert raw Davis Kd (nM) to pKd for regression
        if self.dataset.lower() == 'davis' and self.label_type == 'affinity':
            pkd   = -math.log10(max(float(raw_label), 1e-10) / 1e9)
            label = torch.tensor(pkd, dtype=torch.float32)
        else:
            label = torch.tensor(raw_label, dtype=torch.float32)

        smi = self.lig_dic[self.lig[ind]]
        seq = self.pro_dic[self.pro[ind]]

        # ---- 1. Gating: fetch cached embeddings (NO model forward pass) -----
        drug_tok = self._get_drug_embedding(cid)   # [L_d, H_d]  CPU float32
        prot_tok = self._get_prot_embedding(pid)   # [L_p, H_p]  CPU float32

        batch_dict = {
            'com_id':           self.lig[ind],
            'pro_id':           self.pro[ind],
            'label':            label,
            'shared_drug':      drug_tok,
            'shared_prot':      prot_tok,
            # Padding masks: all False (no padding) — collate_fn updates them
            'shared_drug_mask': torch.zeros(drug_tok.shape[0], dtype=torch.bool),
            'shared_prot_mask': torch.zeros(prot_tok.shape[0], dtype=torch.bool),
        }

        # ---- 2. Expert features (unchanged from original pipeline) ----------

        # 2a. DPDTA — canonical SMILES + standard amino-acid encodings
        smi_enc_can = build_seq_enc(smi, CHARCANSMISET)
        seq_enc_std = build_seq_enc(seq, CHARPROTSET)
        batch_dict['dpdta_com'] = torch.tensor(
            smi_enc_can[:self.MAX_SMI_LEN]
            + [0] * max(0, self.MAX_SMI_LEN - len(smi_enc_can)),
            dtype=torch.long,
        )
        batch_dict['dpdta_pro'] = torch.tensor(
            seq_enc_std[:self.MAX_SEQ_LEN]
            + [0] * max(0, self.MAX_SEQ_LEN - len(seq_enc_std)),
            dtype=torch.long,
        )

        # 2b. GIF-DTI — ISO-SMILES vocabulary
        smi_enc_iso = build_seq_enc(smi, CHARISOSMISET)
        batch_dict['gifdti_com'] = torch.tensor(
            smi_enc_iso[:self.MAX_SMI_LEN]
            + [0] * max(0, self.MAX_SMI_LEN - len(smi_enc_iso)),
            dtype=torch.long,
        )
        batch_dict['gifdti_pro'] = batch_dict['dpdta_pro']
        assert batch_dict['gifdti_com'].max() < 65, "GIFDTI drug index exceeds vocab"
        assert batch_dict['gifdti_pro'].max() < 26, "GIFDTI prot index exceeds vocab"
        batch_dict['gifdti_com_mask'] = (batch_dict['gifdti_com'] == 0)
        batch_dict['gifdti_pro_mask'] = (batch_dict['gifdti_pro'] == 0)

        # 2c. DCDTI — 2048-dim Morgan FP + standard protein tokens
        batch_dict['dcdti_com'] = torch.tensor(self.fp_2048[cid], dtype=torch.float32)
        batch_dict['dcdti_pro'] = batch_dict['dpdta_pro']

        # 2d. MDeePred — 1024-dim FP + 5-channel protein feature matrix
        batch_dict['mdprd_com'] = torch.tensor(self.fp_1024[cid], dtype=torch.float32)
        batch_dict['mdprd_pro'] = self.mdprd_data[pid].to(torch.float32)

        # 2e. PerceiverCPI — graph + Morgan + sequence tokens
        batch_dict['pcpi_morgan']   = torch.tensor(self.fp_1024[cid], dtype=torch.float32)
        batch_dict['pcpi_sequence'] = batch_dict['dpdta_pro']
        batch_dict['pcpi_graph']    = self._get_rdkit_graph(smi)

        # 2f. DeepPurpose — MPNN graph tensors + one-hot protein matrix
        mpnn_val = self.mpnn_data[cid]
        batch_dict['dp_af']  = torch.tensor(mpnn_val[0], dtype=torch.float32)
        batch_dict['dp_bf']  = torch.tensor(mpnn_val[1], dtype=torch.float32)
        batch_dict['dp_ag']  = torch.tensor(mpnn_val[2], dtype=torch.float32)
        batch_dict['dp_bg']  = torch.tensor(mpnn_val[3], dtype=torch.float32)
        batch_dict['dp_abn'] = torch.tensor(mpnn_val[4], dtype=torch.float32)

        dp_pro_tensor = torch.zeros(26, self.MAX_SEQ_LEN, dtype=torch.float32)
        for i, val in enumerate(seq_enc_std[:self.MAX_SEQ_LEN]):
            if val > 0:
                dp_pro_tensor[val, i] = 1.0
        batch_dict['dp_pro'] = dp_pro_tensor

        return batch_dict

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_rdkit_graph(self, smi):
        """Build the atom/bond/adjacency graph tensors for PerceiverCPI."""
        max_a    = self.MAX_SMI_LEN
        pad_atoms = np.zeros((max_a, 5),       dtype=np.float32)
        pad_bonds = np.zeros((max_a, max_a, 3), dtype=np.float32)
        pad_adj   = np.zeros((max_a, max_a),    dtype=np.float32)

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [torch.tensor(pad_atoms),
                    torch.tensor(pad_bonds),
                    torch.tensor(pad_adj)]
        n_a = min(mol.GetNumAtoms(), max_a)
        for idx, atom in enumerate(mol.GetAtoms()):
            if idx >= n_a:
                break
            pad_atoms[idx] = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetNumExplicitHs(),
                int(atom.GetIsAromatic()),
            ]
        adj = np.array(Chem.GetAdjacencyMatrix(mol), dtype=np.float32)
        pad_adj[:n_a, :n_a] = adj[:n_a, :n_a]
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < n_a and j < n_a:
                bf = [
                    bond.GetBondTypeAsDouble(),
                    int(bond.IsInRing()),
                    int(bond.GetIsConjugated()),
                ]
                pad_bonds[i, j] = pad_bonds[j, i] = bf
        return [torch.tensor(pad_atoms),
                torch.tensor(pad_bonds),
                torch.tensor(pad_adj)]

    def _validate_required_feature_files(self):
        missing = []
        for feat_name, fname in _REQUIRED_COM_FILES.items():
            path = os.path.join(self.com_feat_dir, fname)
            if not os.path.exists(path):
                missing.append((feat_name, path))
        for feat_name, fname in _REQUIRED_PRO_FILES.items():
            path = os.path.join(self.pro_feat_dir, fname)
            if not os.path.exists(path):
                missing.append((feat_name, path))
        if missing:
            msg = "\n".join(f"  [{n}] {p}" for n, p in missing)
            raise RuntimeError(
                f"[MoEDataset] Missing required precomputed feature files "
                f"for '{self.dataset}':\n{msg}"
            )

    def validate_features(self):
        """Sanity-check feature array sizes and a few sample embeddings."""
        if len(self.fp_2048) != self.num_lig:
            raise RuntimeError(
                f"[MoEDataset] fps_2048.npy size mismatch: "
                f"{len(self.fp_2048)} vs num_lig={self.num_lig}"
            )
        if len(self.fp_1024) != self.num_lig:
            raise RuntimeError(
                f"[MoEDataset] fps_1024.npy size mismatch: "
                f"{len(self.fp_1024)} vs num_lig={self.num_lig}"
            )
        if len(self.mpnn_data) != self.num_lig:
            raise RuntimeError(
                f"[MoEDataset] dp_mpnn.npy size mismatch: "
                f"{len(self.mpnn_data)} vs num_lig={self.num_lig}"
            )
        if len(self.dp_pro_data) != self.num_pro:
            raise RuntimeError(
                f"[MoEDataset] dp_pro.npy size mismatch: "
                f"{len(self.dp_pro_data)} vs num_pro={self.num_pro}"
            )
        if len(self.mdprd_data) != self.num_pro:
            raise RuntimeError(
                f"[MoEDataset] mdprd_pro.pth size mismatch: "
                f"{len(self.mdprd_data)} vs num_pro={self.num_pro}"
            )

        sample_checks = sorted(set([
            0,
            max(0, self.num_interaction // 2),
            max(0, self.num_interaction - 1),
        ]))
        for ind in sample_checks:
            cid = self.lig_mapping.inverse[self.lig[ind]]
            pid = self.pro_mapping.inverse[self.pro[ind]]

            fp2048 = np.asarray(self.fp_2048[cid])
            fp1024 = np.asarray(self.fp_1024[cid])
            if fp2048.ndim != 1 or fp2048.shape[0] != 2048:
                raise RuntimeError(
                    f"[MoEDataset] Invalid 2048-bit Morgan fingerprint "
                    f"for cid={cid}: shape={fp2048.shape}"
                )
            if fp1024.ndim != 1 or fp1024.shape[0] != 1024:
                raise RuntimeError(
                    f"[MoEDataset] Invalid 1024-bit Morgan fingerprint "
                    f"for cid={cid}: shape={fp1024.shape}"
                )

            # ── Validate cached embeddings for these sample IDs ───────────
            drug_tok = self._get_drug_embedding(cid)
            prot_tok = self._get_prot_embedding(pid)

            if drug_tok.ndim != 2 or drug_tok.shape[0] <= 0 \
                    or drug_tok.shape[1] != self.chembert_hidden_size:
                raise RuntimeError(
                    f"[MoEDataset] Invalid cached ChemBERT embedding for "
                    f"cid={cid}: shape={tuple(drug_tok.shape)}"
                )
            if prot_tok.ndim != 2 or prot_tok.shape[0] <= 0 \
                    or prot_tok.shape[1] != self.esm_hidden_size:
                raise RuntimeError(
                    f"[MoEDataset] Invalid cached ESM embedding for "
                    f"pid={pid}: shape={tuple(prot_tok.shape)}"
                )

            mpnn_val = self.mpnn_data[cid]
            if len(mpnn_val) != 5:
                raise RuntimeError(
                    f"[MoEDataset] Invalid MPNN entry for cid={cid}: "
                    f"expected 5 tensors, got {len(mpnn_val)}"
                )
            if np.asarray(mpnn_val[0]).shape[-1] != 39:
                raise RuntimeError(
                    f"[MoEDataset] Invalid atom feature dim in dp_mpnn.npy "
                    f"for cid={cid}: {np.asarray(mpnn_val[0]).shape}"
                )
            if np.asarray(mpnn_val[1]).shape[-1] != 50:
                raise RuntimeError(
                    f"[MoEDataset] Invalid bond feature dim in dp_mpnn.npy "
                    f"for cid={cid}: {np.asarray(mpnn_val[1]).shape}"
                )

            dp_pro = np.asarray(self.dp_pro_data[pid])
            if dp_pro.ndim != 1 or dp_pro.shape[0] <= 0:
                raise RuntimeError(
                    f"[MoEDataset] Invalid dp protein sequence for "
                    f"pid={pid}: shape={dp_pro.shape}"
                )

            mdprd = self.mdprd_data[pid]
            if tuple(mdprd.shape) != (5, 500, 500):
                raise RuntimeError(
                    f"[MoEDataset] Invalid mdprd protein tensor for "
                    f"pid={pid}: shape={tuple(mdprd.shape)}"
                )


# ---------------------------------------------------------------------------
# Collate function  (unchanged)
# ---------------------------------------------------------------------------

def _pad_tensor_list(tensors, device=None):
    """Pads a list of variable-length tensors to the batch maximum length."""
    if not tensors:
        return None, None
    max_len  = max(t.shape[0] for t in tensors)
    feat_dim = tensors[0].shape[1:] if tensors[0].ndim > 1 else None

    padded, masks = [], []
    for t in tensors:
        curr_len = t.shape[0]
        if curr_len < max_len:
            pad_shape = (max_len - curr_len,) + (feat_dim if feat_dim else ())
            padding   = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
            p = torch.cat([t, padding], dim=0)
            m = torch.cat([
                torch.zeros(curr_len,           dtype=torch.bool, device=t.device),
                torch.ones(max_len - curr_len,  dtype=torch.bool, device=t.device),
            ], dim=0)
        else:
            p = t
            m = torch.zeros(curr_len, dtype=torch.bool, device=t.device)
        padded.append(p)
        masks.append(m)

    return torch.stack(padded, dim=0), torch.stack(masks, dim=0)


def moe_collate_fn(batch):
    collated  = {}
    seq_keys  = ['shared_drug', 'shared_prot']
    graph_keys = ['dp_af', 'dp_bf', 'dp_ag', 'dp_bg', 'dp_abn']

    # Pad variable-length gating embeddings; update masks
    for key in seq_keys:
        if key in batch[0]:
            collated[key], collated[f"{key}_mask"] = _pad_tensor_list(
                [sample[key] for sample in batch]
            )

    # Pad MPNN graph tensors for DeepPurpose
    for key in graph_keys:
        if key in batch[0]:
            collated[key], _ = _pad_tensor_list(
                [sample[key] for sample in batch]
            )

    # Keep PerceiverCPI graph as a list (expert handles its own padding)
    if 'pcpi_graph' in batch[0]:
        collated['pcpi_graph'] = [sample['pcpi_graph'] for sample in batch]

    # Collate everything else with the default collator
    mask_keys      = {f"{k}_mask" for k in seq_keys}
    processed_keys = (set(collated.keys()) | set(seq_keys)
                      | mask_keys | set(graph_keys) | {'pcpi_graph'})
    remaining_keys = [k for k in batch[0].keys() if k not in processed_keys]

    if remaining_keys:
        for key in remaining_keys:
            collated[key] = default_collate([sample[key] for sample in batch])

    return collated


MoEDataset.collate_fn = staticmethod(moe_collate_fn)


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def get_dataset_loader(
    root, dataset_name, batch_size=32, shuffle=True,
    MAX_SMI_LEN=100, MAX_SEQ_LEN=1000, mode='load',
    rebuild_cache=False,
):
    dataset = MoEDataset(
        root, dataset_name,
        MAX_SMI_LEN=MAX_SMI_LEN,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        mode=mode,
        rebuild_cache=rebuild_cache,
    )
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=moe_collate_fn)


def get_davis_loader(root, batch_size=32, shuffle=True):
    return get_dataset_loader(root, 'davis', batch_size, shuffle)


def get_kiba_loader(root, batch_size=32, shuffle=True):
    return get_dataset_loader(root, 'kiba', batch_size, shuffle)


def get_human_loader(root, batch_size=32, shuffle=True):
    return get_dataset_loader(root, 'human', batch_size, shuffle)


def get_kinome_loader(root, batch_size=32, shuffle=True):
    return get_dataset_loader(root, 'kinome', batch_size, shuffle)
