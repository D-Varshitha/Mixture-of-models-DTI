import json
import os

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from config import args
from .dataset import CPIDataset, CHARPROTSET, CHARCANSMISET, CHARISOSMISET, PRO_3GRAM_DICT
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
    def __init__(self, root, dataset_name, MAX_SMI_LEN=100, MAX_SEQ_LEN=1000,
                 label_type='label', mode='load', subset_size=None):
        super(MoEDataset, self).__init__(root, dataset_name, label_type, mode, subset_size=subset_size)

        self.MAX_SMI_LEN = MAX_SMI_LEN
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.mode = mode

        self.shared_emb_dir = os.path.join(self.root, 'dataset', self.dataset, 'shared_emb')
        self.drug_emb_dir = os.path.join(self.shared_emb_dir, 'drug')
        self.prot_emb_dir = os.path.join(self.shared_emb_dir, 'protein')
        self.shared_meta_path = os.path.join(self.shared_emb_dir, 'metadata.json')
        self._drug_emb_cache = {}
        self._prot_emb_cache = {}

        if mode == 'generate':
            print(f"[MoEDataset] Generating precomputed expert features for '{dataset_name}'...")
            os.makedirs(self.com_feat_dir, exist_ok=True)
            os.makedirs(self.pro_feat_dir, exist_ok=True)
            self.process_molecule(MAX_SMI_LEN, 'fp_2048')
            self.process_molecule(MAX_SMI_LEN, 'fp_1024')
            self.process_molecule(MAX_SMI_LEN, 'mpnn')
            self.process_protein(MAX_SEQ_LEN, 'mdprd')
            self.process_protein(MAX_SEQ_LEN, 'dp')
            self._generate_shared_embeddings()

        self._validate_required_feature_files()
        self.fp_2048 = self.get_fp(2048)
        self.fp_1024 = self.get_fp(1024)
        self.mpnn_data = self.get_mpnn_feature
        self.dp_pro_data = self.get_dp
        self.mdprd_data = self.get_mdprd

        self._validate_shared_embeddings()
        self.validate_features()

    def __len__(self):
        return self.num_interaction

    def _shared_drug_path(self, cid: int) -> str:
        return os.path.join(self.drug_emb_dir, f'drug_{cid}.pt')

    def _shared_prot_path(self, pid: int) -> str:
        return os.path.join(self.prot_emb_dir, f'protein_{pid}.pt')

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
                f"[MoEDataset] Missing required precomputed feature files for '{self.dataset}':\n{msg}"
            )

    def _generate_shared_embeddings(self):
        generator = PretrainedEmbeddingGenerator(
            esm_model_name=args.esm_model_name,
            chembert_model_name=args.chembert_model_name,
            cache_dir=args.hf_cache_dir,
            device='cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu',
            protein_chunk_len=args.protein_chunk_len,
            protein_chunk_stride=args.protein_chunk_stride,
        )
        drug_texts = [self.lig_dic[self.lig_mapping[idx]] for idx in range(self.num_lig)]
        prot_texts = [self.pro_dic[self.pro_mapping[idx]] for idx in range(self.num_pro)]
        drug_paths = [self._shared_drug_path(cid) for cid in range(self.num_lig)]
        prot_paths = [self._shared_prot_path(pid) for pid in range(self.num_pro)]
        generator.generate_and_save(
            drug_texts=drug_texts,
            prot_texts=prot_texts,
            drug_paths=drug_paths,
            prot_paths=prot_paths,
            metadata_path=self.shared_meta_path,
        )

    def _validate_shared_embeddings(self):
        if not os.path.exists(self.shared_meta_path):
            raise RuntimeError(
                f"[MoEDataset] Missing shared embedding metadata: {self.shared_meta_path}. "
                "Run with --get-dataset generate to build real pretrained ESM/ChemBERT embeddings."
            )
        with open(self.shared_meta_path, "r", encoding="utf-8") as f:
            self.shared_meta = json.load(f)

        required_meta = [
            "esm_model_name", "chembert_model_name",
            "esm_hidden_size", "chembert_hidden_size",
            "num_drugs", "num_proteins",
        ]
        missing_meta = [k for k in required_meta if k not in self.shared_meta]
        if missing_meta:
            raise RuntimeError(f"[MoEDataset] Shared embedding metadata missing keys: {missing_meta}")

        if int(self.shared_meta["num_drugs"]) != self.num_lig:
            raise RuntimeError(
                f"[MoEDataset] Shared drug embedding count mismatch: "
                f"{self.shared_meta['num_drugs']} vs num_lig={self.num_lig}"
            )
        if int(self.shared_meta["num_proteins"]) != self.num_pro:
            raise RuntimeError(
                f"[MoEDataset] Shared protein embedding count mismatch: "
                f"{self.shared_meta['num_proteins']} vs num_pro={self.num_pro}"
            )
        if self.shared_meta["esm_model_name"] != args.esm_model_name:
            raise RuntimeError(
                f"[MoEDataset] Shared ESM cache mismatch: "
                f"{self.shared_meta['esm_model_name']} vs requested {args.esm_model_name}"
            )
        if self.shared_meta["chembert_model_name"] != args.chembert_model_name:
            raise RuntimeError(
                f"[MoEDataset] Shared ChemBERT cache mismatch: "
                f"{self.shared_meta['chembert_model_name']} vs requested {args.chembert_model_name}"
            )

        missing_files = []
        for cid in range(self.num_lig):
            if not os.path.exists(self._shared_drug_path(cid)):
                missing_files.append(self._shared_drug_path(cid))
        for pid in range(self.num_pro):
            if not os.path.exists(self._shared_prot_path(pid)):
                missing_files.append(self._shared_prot_path(pid))
        if missing_files:
            raise RuntimeError(
                f"[MoEDataset] Missing cached shared embedding files. First missing path: {missing_files[0]}"
            )

        self.chembert_hidden_size = int(self.shared_meta["chembert_hidden_size"])
        self.esm_hidden_size = int(self.shared_meta["esm_hidden_size"])

    def _load_shared_embedding(self, kind: str, idx: int) -> torch.Tensor:
        cache = self._drug_emb_cache if kind == 'drug' else self._prot_emb_cache
        if idx not in cache:
            path = self._shared_drug_path(idx) if kind == 'drug' else self._shared_prot_path(idx)
            payload = torch.load(path, map_location='cpu')
            emb = payload["embedding"] if isinstance(payload, dict) else payload
            if not isinstance(emb, torch.Tensor):
                raise RuntimeError(f"[MoEDataset] Shared embedding payload must be a tensor at {path}")
            cache[idx] = emb.to(torch.float32)
        return cache[idx]

    def __getitem__(self, ind):
        cid = self.lig_mapping.inverse[self.lig[ind]]
        pid = self.pro_mapping.inverse[self.pro[ind]]
        label = torch.tensor(self.label[ind], dtype=torch.float32)

        smi = self.lig_dic[self.lig[ind]]
        seq = self.pro_dic[self.pro[ind]]

        # ---- 1. Gating Network: Uses SHARED precomputed embeddings ----
        drug_tok = self._load_shared_embedding('drug', cid)
        prot_tok = self._load_shared_embedding('protein', pid)
        
        batch_dict = {
            'com_id': self.lig[ind],
            'pro_id': self.pro[ind],
            'label': label,
            'shared_drug': drug_tok,
            'shared_prot': prot_tok,
            # Masks initialized as zeros (fully unmasked); collate_fn will pad and set masks correctly.
            'shared_drug_mask': torch.zeros(drug_tok.shape[0], dtype=torch.bool),
            'shared_prot_mask': torch.zeros(prot_tok.shape[0], dtype=torch.bool),
        }

        # ---- 2. Experts: Use their OWN dedicated features/encoders ----
        # 2a. Expert: DPDTA (Standard SMI/PRO Encodings)
        smi_enc_can = build_seq_enc(smi, CHARCANSMISET)
        seq_enc_std = build_seq_enc(seq, CHARPROTSET)
        
        batch_dict['dpdta_com'] = torch.tensor(smi_enc_can[:100] + [0]*max(0, 100-len(smi_enc_can)), dtype=torch.long)
        batch_dict['dpdta_pro'] = torch.tensor(seq_enc_std[:1000] + [0]*max(0, 1000-len(seq_enc_std)), dtype=torch.long)

        # 2b. Expert: GIF-DTI (Requires ISO-SMILES vocabulary)
        smi_enc_iso = build_seq_enc(smi, CHARISOSMISET)
        batch_dict['gifdti_com'] = torch.tensor(smi_enc_iso[:100] + [0]*max(0, 100-len(smi_enc_iso)), dtype=torch.long)
        batch_dict['gifdti_pro'] = batch_dict['dpdta_pro'] # uses standard PROTSET
        
        # Safety checks for Embedding sanity
        assert batch_dict['gifdti_com'].max() < 65, "GIFDTI drug index exceeds vocab"
        assert batch_dict['gifdti_pro'].max() < 26, "GIFDTI prot index exceeds vocab"
        batch_dict['gifdti_com_mask'] = (batch_dict['gifdti_com'] == 0)
        batch_dict['gifdti_pro_mask'] = (batch_dict['gifdti_pro'] == 0)

        # 2c. Expert: DCDTI (Requires Fingerprints + Protein Tokens)
        batch_dict['dcdti_com'] = torch.tensor(self.fp_2048[cid], dtype=torch.float32)
        # DeepConvDTI expects standard AA tokens (indices 0-25) at current construction (2500 vocab)
        batch_dict['dcdti_pro'] = batch_dict['dpdta_pro']

        # 2d. Expert: MDeePred (Req. Fingerprints + Specific Bio-Feature Matrix)
        batch_dict['mdprd_com'] = torch.tensor(self.fp_1024[cid], dtype=torch.float32)
        batch_dict['mdprd_pro'] = self.mdprd_data[pid].to(torch.float32)

        # 2e. Expert: PerceiverCPI (Req. Graph + Morgan + Sequence Tokens)
        batch_dict['pcpi_morgan']   = torch.tensor(self.fp_1024[cid], dtype=torch.float32)
        batch_dict['pcpi_sequence'] = batch_dict['dpdta_pro']
        batch_dict['pcpi_graph']    = self._get_rdkit_graph(smi)

        # 2f. Expert: DeepPurpose (Req. MPNN Graph + Protein One-Hot Matrix)
        mpnn_val = self.mpnn_data[cid]
        batch_dict['dp_af']  = torch.tensor(mpnn_val[0], dtype=torch.float32)
        batch_dict['dp_bf']  = torch.tensor(mpnn_val[1], dtype=torch.float32)
        batch_dict['dp_ag']  = torch.tensor(mpnn_val[2], dtype=torch.float32)
        batch_dict['dp_bg']  = torch.tensor(mpnn_val[3], dtype=torch.float32)
        batch_dict['dp_abn'] = torch.tensor(mpnn_val[4], dtype=torch.float32)
        
        # DeepPurpose CNN encoder expects [26, L] one-hot matrix
        dp_pro = build_one_hot_enc(seq, 1000, CHARPROTSET) # [1000, 25] -> we need [26, 1000]
        # Pad/reshaping to [26, 1000] (0 for padding, 1-25 for AAs)
        dp_pro_tensor = torch.zeros(26, 1000, dtype=torch.float32)
        for i, val in enumerate(seq_enc_std[:1000]):
            dp_pro_tensor[val, i] = 1.0
        batch_dict['dp_pro'] = dp_pro_tensor

        return batch_dict

    def _get_rdkit_graph(self, smi):
        """Helper to generate internal graph for PerceiverCPI/MPNN."""
        max_a = 100
        pad_atoms = np.zeros((max_a, 5), dtype=np.float32)
        pad_bonds = np.zeros((max_a, max_a, 3), dtype=np.float32)
        pad_adj = np.zeros((max_a, max_a), dtype=np.float32)

        mol = Chem.MolFromSmiles(smi)
        if mol is None: return [torch.tensor(pad_atoms), torch.tensor(pad_bonds), torch.tensor(pad_adj)]
        n_a = min(mol.GetNumAtoms(), max_a)
        for idx, atom in enumerate(mol.GetAtoms()):
            if idx >= n_a: break
            pad_atoms[idx] = [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(), 
                              atom.GetNumExplicitHs(), int(atom.GetIsAromatic())]
        adj = np.array(Chem.GetAdjacencyMatrix(mol), dtype=np.float32)
        pad_adj[:n_a, :n_a] = adj[:n_a, :n_a]
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < n_a and j < n_a:
                bf = [bond.GetBondTypeAsDouble(), int(bond.IsInRing()), int(bond.GetIsConjugated())]
                pad_bonds[i, j] = pad_bonds[j, i] = bf
        return [torch.tensor(pad_atoms), torch.tensor(pad_bonds), torch.tensor(pad_adj)]

    def validate_features(self):
        if len(self.fp_2048) != self.num_lig:
            raise RuntimeError(f"[MoEDataset] fps_2048.npy size mismatch: {len(self.fp_2048)} vs num_lig={self.num_lig}")
        if len(self.fp_1024) != self.num_lig:
            raise RuntimeError(f"[MoEDataset] fps_1024.npy size mismatch: {len(self.fp_1024)} vs num_lig={self.num_lig}")
        if len(self.mpnn_data) != self.num_lig:
            raise RuntimeError(f"[MoEDataset] dp_mpnn.npy size mismatch: {len(self.mpnn_data)} vs num_lig={self.num_lig}")
        if len(self.dp_pro_data) != self.num_pro:
            raise RuntimeError(f"[MoEDataset] dp_pro.npy size mismatch: {len(self.dp_pro_data)} vs num_pro={self.num_pro}")
        if len(self.mdprd_data) != self.num_pro:
            raise RuntimeError(f"[MoEDataset] mdprd_pro.pth size mismatch: {len(self.mdprd_data)} vs num_pro={self.num_pro}")

        sample_checks = sorted(set([0, max(0, self.num_interaction // 2), max(0, self.num_interaction - 1)]))
        for ind in sample_checks:
            cid = self.lig_mapping.inverse[self.lig[ind]]
            pid = self.pro_mapping.inverse[self.pro[ind]]

            fp2048 = np.asarray(self.fp_2048[cid])
            fp1024 = np.asarray(self.fp_1024[cid])
            if fp2048.ndim != 1 or fp2048.shape[0] != 2048:
                raise RuntimeError(f"[MoEDataset] Invalid 2048-bit Morgan fingerprint for cid={cid}: shape={fp2048.shape}")
            if fp1024.ndim != 1 or fp1024.shape[0] != 1024:
                raise RuntimeError(f"[MoEDataset] Invalid 1024-bit Morgan fingerprint for cid={cid}: shape={fp1024.shape}")

            drug_tok = self._load_shared_embedding('drug', cid)
            prot_tok = self._load_shared_embedding('protein', pid)
            if drug_tok.ndim != 2 or drug_tok.shape[0] <= 0 or drug_tok.shape[1] != self.chembert_hidden_size:
                raise RuntimeError(f"[MoEDataset] Invalid ChemBERT token embeddings for cid={cid}: shape={tuple(drug_tok.shape)}")
            if prot_tok.ndim != 2 or prot_tok.shape[0] <= 0 or prot_tok.shape[1] != self.esm_hidden_size:
                raise RuntimeError(f"[MoEDataset] Invalid ESM token embeddings for pid={pid}: shape={tuple(prot_tok.shape)}")

            mpnn_val = self.mpnn_data[cid]
            if len(mpnn_val) != 5:
                raise RuntimeError(f"[MoEDataset] Invalid MPNN entry for cid={cid}: expected 5 tensors, got {len(mpnn_val)}")
            if np.asarray(mpnn_val[0]).shape[-1] != 39:
                raise RuntimeError(f"[MoEDataset] Invalid atom feature dim in dp_mpnn.npy for cid={cid}: {np.asarray(mpnn_val[0]).shape}")
            if np.asarray(mpnn_val[1]).shape[-1] != 50:
                raise RuntimeError(f"[MoEDataset] Invalid bond feature dim in dp_mpnn.npy for cid={cid}: {np.asarray(mpnn_val[1]).shape}")

            dp_pro = np.asarray(self.dp_pro_data[pid])
            if dp_pro.ndim != 1 or dp_pro.shape[0] <= 0:
                raise RuntimeError(f"[MoEDataset] Invalid dp protein sequence for pid={pid}: shape={dp_pro.shape}")

            mdprd = self.mdprd_data[pid]
            if tuple(mdprd.shape) != (5, 500, 500):
                raise RuntimeError(f"[MoEDataset] Invalid mdprd protein tensor for pid={pid}: shape={tuple(mdprd.shape)}")


def _pad_tensor_list(tensors, device=None):
    """Pads a list of tensors to the batch maximum length."""
    if not tensors: return None, None
    max_len = max(t.shape[0] for t in tensors)
    feat_dim = tensors[0].shape[1:] if tensors[0].ndim > 1 else None
    
    padded = []
    masks = []
    for t in tensors:
        curr_len = t.shape[0]
        if curr_len < max_len:
            # Create padding
            pad_shape = (max_len - curr_len,) + (feat_dim if feat_dim else ())
            padding = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
            p = torch.cat([t, padding], dim=0)
            
            # Create mask (True for padded positions)
            m = torch.cat([
                torch.zeros(curr_len, dtype=torch.bool, device=t.device),
                torch.ones(max_len - curr_len, dtype=torch.bool, device=t.device)
            ], dim=0)
            
            padded.append(p)
            masks.append(m)
        else:
            padded.append(t)
            masks.append(torch.zeros(curr_len, dtype=torch.bool, device=t.device))
            
    return torch.stack(padded, dim=0), torch.stack(masks, dim=0)

def moe_collate_fn(batch):
    """
    Robust collate_fn that pads all variable-length sequence features 
    and handles expert-specific inputs.
    """
    collated = {}
    
    # 1. Define groups of keys by their padding requirements
    # Keys that need [PaddedTensor, Mask] return
    seq_keys = ['shared_drug', 'shared_prot', 'gifdti_com', 'gifdti_pro']
    
    # Experts that use lists of (padded) atom/bond features
    graph_keys = ['dp_af', 'dp_bf', 'dp_ag', 'dp_bg', 'dp_abn']
    
    # Fixed-length keys or scalars handled by default_collate
    # (label, ids, fingerprints, mdprd_pro, pcpi_morgan, etc.)

    # 2. Extract and Pad Sequence/Expert Features
    for key in seq_keys:
        if key in batch[0]:
            collated[key], collated[f"{key}_mask"] = _pad_tensor_list([sample[key] for sample in batch])
            
    for key in graph_keys:
        if key in batch[0]:
            # We pad these for the Experts to receive fixed-shape tensors
            collated[key], _ = _pad_tensor_list([sample[key] for sample in batch])

    # 3. Handle pcpi_graph (list of 3 tensors) separately
    if 'pcpi_graph' in batch[0]:
        # Keep as list of lists for expert-level handled padding (or pack here)
        collated['pcpi_graph'] = [sample['pcpi_graph'] for sample in batch]

    # 4. Handle everything else with default_collate
    processed_keys = set(collated.keys()) | set(seq_keys) | set(graph_keys) | {'pcpi_graph'}
    remaining_keys = [k for k in batch[0].keys() if k not in processed_keys]
    
    if remaining_keys:
        for key in remaining_keys:
            collated[key] = default_collate([sample[key] for sample in batch])
            
    return collated


MoEDataset.collate_fn = staticmethod(moe_collate_fn)


def get_dataset_loader(root, dataset_name, batch_size=32, shuffle=True,
                       MAX_SMI_LEN=100, MAX_SEQ_LEN=1000, mode='load'):
    dataset = MoEDataset(
        root, dataset_name,
        MAX_SMI_LEN=MAX_SMI_LEN,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        mode=mode,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=moe_collate_fn)


def get_davis_loader(root, batch_size=32, shuffle=True):
    return get_dataset_loader(root, 'davis', batch_size, shuffle)


def get_kiba_loader(root, batch_size=32, shuffle=True):
    return get_dataset_loader(root, 'kiba', batch_size, shuffle)


def get_human_loader(root, batch_size=32, shuffle=True):
    return get_dataset_loader(root, 'human', batch_size, shuffle)


def get_kinome_loader(root, batch_size=32, shuffle=True):
    return get_dataset_loader(root, 'kinome', batch_size, shuffle)
