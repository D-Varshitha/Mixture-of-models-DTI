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
        drug_chunk_len: int = 510,
        drug_chunk_stride: int = 255,
    ):
        self.esm_model_name = esm_model_name
        self.chembert_model_name = chembert_model_name
        self.cache_dir = cache_dir
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.protein_chunk_len = protein_chunk_len
        self.protein_chunk_stride = protein_chunk_stride
        self.drug_chunk_len = drug_chunk_len   
        self.drug_chunk_stride = drug_chunk_stride 

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
        """
        Generates token-level ChemBERT embeddings with sliding-window chunking.

        SMILES must be chunked at TOKEN level (not character level) because
        multi-char tokens like Cl, Br, [NH], @@ are single tokens —
        character slicing would corrupt them.

        Strategy:
          1. Tokenize full SMILES *without* special tokens → raw token IDs.
          2. If total tokens <= drug_chunk_len  →  fast single-pass (common case).
          3. Otherwise slide a window of size drug_chunk_len with stride drug_chunk_stride
             over the token IDs, prepend/append [CLS]/[EOS] manually, run the model,
             strip specials, and overlap-average each token position.
        Returns:
            [num_tokens, hidden_size]  float32 on CPU
        """
        self._load_chembert()
        if not smiles:
            raise RuntimeError("Cannot embed empty SMILES string with ChemBERT.")

        # ── Step 1: tokenize without special tokens to get raw IDs ──────────
        raw = self._chem_tokenizer(
            smiles,
            add_special_tokens=False,
            return_tensors="pt",
        )
        all_ids = raw["input_ids"][0]          
        total_tokens = all_ids.size(0)

        chunk_len = max(1, self.drug_chunk_len)
        stride    = max(1, min(self.drug_chunk_stride, chunk_len))

        cls_id = self._chem_tokenizer.cls_token_id
        eos_id = (self._chem_tokenizer.eos_token_id
                  or self._chem_tokenizer.sep_token_id)

        # ── Step 2: fast path — fits in one chunk ────────────────────────────
        if total_tokens <= chunk_len:
            chunk_with_sp = torch.cat([
                torch.tensor([cls_id], dtype=torch.long),
                all_ids,
                torch.tensor([eos_id], dtype=torch.long),
            ]).unsqueeze(0).to(self.device)           
            attn = torch.ones_like(chunk_with_sp)
            with torch.no_grad():
                out = self._chem_model(input_ids=chunk_with_sp,
                                       attention_mask=attn)
            token_embs = out.last_hidden_state.squeeze(0)[1:-1]  
            if token_embs.size(0) == 0:
                raise RuntimeError(
                    f"ChemBERT returned zero token embeddings for SMILES {smiles!r}")
            return token_embs.cpu()

        # ── Step 3: sliding-window path (long SMILES) ────────────────────────
        hidden_size = self.chembert_hidden_size          
        full_embed = torch.zeros(total_tokens, hidden_size, device=self.device)
        counts     = torch.zeros(total_tokens,              device=self.device)

        for start in range(0, total_tokens, stride):
            end        = min(start + chunk_len, total_tokens)
            chunk_ids  = all_ids[start:end]

            chunk_with_sp = torch.cat([
                torch.tensor([cls_id], dtype=torch.long),
                chunk_ids,
                torch.tensor([eos_id], dtype=torch.long),
            ]).unsqueeze(0).to(self.device)           
            attn = torch.ones_like(chunk_with_sp)

            with torch.no_grad():
                out = self._chem_model(input_ids=chunk_with_sp,
                                       attention_mask=attn)

            # Strip [CLS] (pos 0) and [EOS] (pos -1)
            token_embs  = out.last_hidden_state.squeeze(0)[1:-1]  # [chunk_len, H]
            chunk_tokens = token_embs.size(0)
            if chunk_tokens == 0:
                raise RuntimeError(
                    f"ChemBERT returned zero embeddings for SMILES chunk "
                    f"starting at token {start}: {smiles!r}")

            full_embed[start:start + chunk_tokens] += token_embs
            counts    [start:start + chunk_tokens] += 1

            if end == total_tokens:
                break

        if (counts == 0).any():
            missing = int((counts == 0).sum().item())
            raise RuntimeError(
                f"Drug overlap embedding left {missing} token positions uncovered "
                f"for SMILES: {smiles!r}")

        return (full_embed / counts.unsqueeze(1)).cpu()


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
            "drug_chunk_len": self.drug_chunk_len,
            "drug_chunk_stride": self.drug_chunk_stride,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
