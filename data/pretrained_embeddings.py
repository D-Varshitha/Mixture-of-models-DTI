import json
import os
from typing import Dict, List

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise RuntimeError(
        "transformers is required for real pretrained ESM/ChemBERT embeddings. "
        "Install it in the active environment before running MoE training."
    ) from exc


class PretrainedEmbeddingGenerator:
    def __init__(
        self,
        esm_model_name: str,
        chembert_model_name: str,
        cache_dir: str = None,
        device: str = None,
        protein_chunk_len: int = 1022,
        protein_chunk_stride: int = 512,
    ):
        self.esm_model_name = esm_model_name
        self.chembert_model_name = chembert_model_name
        self.cache_dir = cache_dir
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.protein_chunk_len = protein_chunk_len
        self.protein_chunk_stride = protein_chunk_stride

        self._esm_tokenizer = None
        self._esm_model = None
        self._chem_tokenizer = None
        self._chem_model = None

    def _load_esm(self):
        if self._esm_model is None:
            self._esm_tokenizer = AutoTokenizer.from_pretrained(
                self.esm_model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            self._esm_model = AutoModel.from_pretrained(
                self.esm_model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            ).to(self.device)
            self._esm_model.eval()

    def _load_chembert(self):
        if self._chem_model is None:
            self._chem_tokenizer = AutoTokenizer.from_pretrained(
                self.chembert_model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            self._chem_model = AutoModel.from_pretrained(
                self.chembert_model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            ).to(self.device)
            self._chem_model.eval()

    @property
    def esm_hidden_size(self) -> int:
        self._load_esm()
        return int(self._esm_model.config.hidden_size)

    @property
    def chembert_hidden_size(self) -> int:
        self._load_chembert()
        return int(self._chem_model.config.hidden_size)

    def embed_drug(self, smiles: str) -> torch.Tensor:
        self._load_chembert()
        inputs = self._chem_tokenizer(
            smiles,
            return_tensors="pt",
            add_special_tokens=True,
            return_special_tokens_mask=True,
            truncation=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._chem_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        token_embs = outputs.last_hidden_state.squeeze(0)
        special_mask = inputs["special_tokens_mask"].squeeze(0).bool()
        attn_mask = inputs["attention_mask"].squeeze(0).bool()
        keep = attn_mask & (~special_mask)
        token_embs = token_embs[keep]
        if token_embs.ndim != 2 or token_embs.size(0) == 0:
            raise RuntimeError(f"ChemBERT returned invalid token embeddings for SMILES {smiles!r}")
        return token_embs.cpu()

    def embed_protein(self, seq: str) -> torch.Tensor:
        self._load_esm()
        hidden_size = self.esm_hidden_size
        seq_len = len(seq)
        if seq_len == 0:
            raise RuntimeError("Cannot embed empty protein sequence with ESM.")

        full_embed = torch.zeros(seq_len, hidden_size, device=self.device)
        counts = torch.zeros(seq_len, device=self.device)
        chunk_len = max(1, self.protein_chunk_len)
        stride = max(1, min(self.protein_chunk_stride, chunk_len))

        for start in range(0, seq_len, stride):
            end = min(start + chunk_len, seq_len)
            chunk = seq[start:end]
            inputs = self._esm_tokenizer(
                chunk,
                return_tensors="pt",
                add_special_tokens=True,
                return_special_tokens_mask=True,
                truncation=False,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._esm_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

            token_embs = outputs.last_hidden_state.squeeze(0)
            special_mask = inputs["special_tokens_mask"].squeeze(0).bool()
            attn_mask = inputs["attention_mask"].squeeze(0).bool()
            keep = attn_mask & (~special_mask)
            token_embs = token_embs[keep]
            chunk_tokens = token_embs.size(0)
            if chunk_tokens == 0:
                raise RuntimeError(f"ESM returned zero token embeddings for sequence chunk starting at {start}.")

            full_embed[start:start + chunk_tokens] += token_embs
            counts[start:start + chunk_tokens] += 1

            if end == seq_len:
                break

        if (counts == 0).any():
            missing = int((counts == 0).sum().item())
            raise RuntimeError(f"ESM overlap embedding left {missing} protein positions uncovered.")

        return (full_embed / counts.unsqueeze(1)).cpu()

    def generate_and_save(
        self,
        drug_texts: List[str],
        prot_texts: List[str],
        drug_paths: List[str],
        prot_paths: List[str],
        metadata_path: str,
    ):
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        for path in drug_paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        for path in prot_paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        for idx, (text, path) in enumerate(zip(drug_texts, drug_paths)):
            if not os.path.exists(path):
                emb = self.embed_drug(text)
                torch.save({"embedding": emb}, path)
            if idx % 100 == 0:
                print(f"[SharedEmb] Processed drug {idx}/{len(drug_paths)}")

        for idx, (text, path) in enumerate(zip(prot_texts, prot_paths)):
            if not os.path.exists(path):
                emb = self.embed_protein(text)
                torch.save({"embedding": emb}, path)
            if idx % 50 == 0:
                print(f"[SharedEmb] Processed protein {idx}/{len(prot_paths)}")

        metadata: Dict[str, object] = {
            "esm_model_name": self.esm_model_name,
            "chembert_model_name": self.chembert_model_name,
            "esm_hidden_size": self.esm_hidden_size,
            "chembert_hidden_size": self.chembert_hidden_size,
            "num_drugs": len(drug_paths),
            "num_proteins": len(prot_paths),
            "protein_chunk_len": self.protein_chunk_len,
            "protein_chunk_stride": self.protein_chunk_stride,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
