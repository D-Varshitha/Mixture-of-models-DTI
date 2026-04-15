"""
DTI Sparse Mixture-of-Experts (MoE) Model
==========================================

Architecture (matches specification):
  1.  Token-level embeddings from SharedGatingEncoder (NO pooling before gating).
  2.  Gating network pools token sequences → MLP → softmax over experts.
  3.  Top-k (k=2) sparse routing.
  4.  Each selected expert runs independently on its own input format.
  5.  Expert scalar predictions are collected into a [B, num_experts] matrix.
  6.  A trainable MLP aggregates expert outputs into the final scalar.
  7.  Auxiliary load-balancing loss (Switch-Transformer style) prevents collapse.

Key fixes over previous version
---------------------------------
* SharedGatingEncoder.forward() now returns token-level tensor [B, L, D]
  (chunked path returns [B, num_chunks, D]) – gating reads token embeddings,
  NOT pooled vectors.
* SharedGatingNetwork now separately pools the token-level tensors for MLP input,
  keeping the distinction between "token-level" and "pooled" clearly separated.
* PerceiverCPI is dispatched correctly: integer token IDs are passed directly
  (no extra unsqueeze).
* Final aggregation uses a trainable MLP (FinalAggMLP), not a simple weighted sum.
* The MLP's weights update during training via standard back-prop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pe', self._build_pe(max_len))

    def _build_pe(self, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def ensure_length(self, target_len: int, device: torch.device):
        if target_len <= self.pe.size(0):
            return
        new_pe = self._build_pe(target_len).to(device=device)
        self.pe = new_pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        self.ensure_length(x.size(1), x.device)
        return x + self.pe[:x.size(1), :]


# ---------------------------------------------------------------------------
# Shared Gating Encoder  –  returns TOKEN-LEVEL embeddings
# ---------------------------------------------------------------------------
class SharedGatingEncoder(nn.Module):
    """
    Encodes an integer token sequence into token-level hidden states [B, L, D].

    For long protein sequences (is_chunked=True, L > chunk_size) the output is
    chunked into [B, num_chunks, D] where each chunk is a mean-pooled chunk
    representation.  This lets the caller see the chunk-level token structure
    rather than a single pooled vector.

    IMPORTANT: This module intentionally does NOT pool to [B, D].
    Pooling is the caller's responsibility so that token-level information is
    preserved for the gating network.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int  = 128,
        nhead:      int  = 4,
        num_layers: int  = 2,
        max_len:    int  = 2000,
        is_chunked: bool = False,
        chunk_size: int  = 512,
        pretrained_dim: int = 640,  # e.g., 640 for ESM2-150M, 768 for ChemBERTa
    ):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pretrained_proj = nn.LazyLinear(d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model * 2, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.is_chunked  = is_chunked
        self.chunk_size  = chunk_size

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        # If float, we assume these are pretrained dense features (e.g. ESM/ChemBERT)
        if x.is_floating_point():
            return self.pretrained_proj(x)
        # Otherwise assume integer token IDs
        return self.embedding(x.long())

    def forward(
        self,
        x:    torch.Tensor,          # [B, L] integer token IDs
        mask: torch.Tensor = None,  # [B, L] bool, True = padding
    ) -> torch.Tensor:
        """
        Returns:
            token_embs : [B, L, D]            (standard path)
                       | [B, num_chunks, D]   (chunked path – chunk-level reps)
        """
        if self.is_chunked and x.size(1) > self.chunk_size:
            # ---- Chunked path ------------------------------------------------
            B, L = x.shape[:2]
            num_chunks  = math.ceil(L / self.chunk_size)
            chunk_outputs  = []

            for i in range(num_chunks):
                start      = i * self.chunk_size
                end        = min(start + self.chunk_size, L)
                x_chunk    = x[:, start:end]                       # [B, cs] or [B, cs, D]
                mask_chunk = mask[:, start:end] if mask is not None else None
                original_mask_chunk = mask_chunk

                emb = self._get_embeddings(x_chunk)
                self.pos_encoder.ensure_length(end, emb.device)
                # FIX 8: Use absolute position offset so tokens in chunk i
                # get positions [start, start+chunk_len), not [0, chunk_len).
                # This preserves global positional context across chunk boundaries.
                chunk_len = emb.size(1)
                emb = emb + self.pos_encoder.pe[start:start + chunk_len, :].unsqueeze(0)
                
                # Prevent PyTorch TransformerEncoder crash if a row is entirely padded
                if mask_chunk is not None:
                    mask_chunk = mask_chunk.clone()
                    all_pad_rows = mask_chunk.all(dim=1)
                    if all_pad_rows.any():
                        mask_chunk[all_pad_rows, 0] = False
                        
                out = self.transformer(emb, src_key_padding_mask=mask_chunk)
                if original_mask_chunk is not None:
                    out = out.masked_fill(original_mask_chunk.unsqueeze(-1), 0.0)

                chunk_outputs.append(out)

            return torch.cat(chunk_outputs, dim=1)

        else:
            # ---- Standard path  – full token-level output ---------------------
            emb = self._get_embeddings(x)
            emb = self.pos_encoder(emb)
            # Returns token-level hidden states: [B, L, D]
            original_mask = mask
            if mask is not None:
                mask = mask.clone()
                all_pad_rows = mask.all(dim=1)
                if all_pad_rows.any():
                    mask[all_pad_rows, 0] = False
            out = self.transformer(emb, src_key_padding_mask=mask)
            if original_mask is not None:
                out = out.masked_fill(original_mask.unsqueeze(-1), 0.0)
            return out  # [B, L, D]  ← NO POOLING HERE


# ---------------------------------------------------------------------------
# Attention Pool
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, token_embs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # token_embs: [B, L, D]
        # mask: [B, L] bool, True indicates padding
        
        scores = self.attn_proj(token_embs).squeeze(-1) # [B, L]
        if mask is not None and mask.shape[1] == scores.shape[1]:
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1) # [B, L]
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0) # Handle all-pad edge case
        
        rep = torch.sum(attn_weights.unsqueeze(-1) * token_embs, dim=1) # [B, D]
        return rep


# ---------------------------------------------------------------------------
# Shared Gating Network
# ---------------------------------------------------------------------------
class SharedGatingNetwork(nn.Module):
    """
    Takes token-level embeddings from both drug and protein encoders,
    pools them into fixed-size vectors, then uses an MLP to produce expert
    routing logits.

    The separation is deliberate:
      * SharedGatingEncoder produces token-level tensors  (structural information)
      * SharedGatingNetwork pools + projects those tensors (routing decision)
    """

    def __init__(
        self,
        drug_vocab:  int,
        prot_vocab:  int,
        d_model:     int = 128,
        num_experts: int = 6,
        drug_pretrained_dim: int = 768, # e.g. ChemBERT
        prot_pretrained_dim: int = 640, # e.g. ESM2
    ):
        super().__init__()
        self.drug_enc = SharedGatingEncoder(drug_vocab, d_model, is_chunked=False, pretrained_dim=drug_pretrained_dim)
        self.prot_enc = SharedGatingEncoder(prot_vocab, d_model, is_chunked=True, chunk_size=512, pretrained_dim=prot_pretrained_dim)

        self.drug_pool = AttentionPool(d_model)
        self.prot_pool = AttentionPool(d_model)

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_experts),
        )

    # pool methods removed, replaced by AttentionPool usage in forward

    def forward(
        self,
        drug_tokens: torch.Tensor,
        prot_tokens: torch.Tensor,
        drug_mask:   torch.Tensor = None,
        prot_mask:   torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns routing logits [B, num_experts].
        """
        # 1. Encode to token-level (no pooling inside encoders)
        drug_token_embs = self.drug_enc(drug_tokens, drug_mask)  # [B, L_d, D]
        prot_chunk_embs = self.prot_enc(prot_tokens, prot_mask)  # [B, L_p, D]

        # 2. Pool to fixed-size vectors for the MLP using Attention Pooling
        d_rep = self.drug_pool(drug_token_embs, drug_mask)           # [B, D]
        p_rep = self.prot_pool(prot_chunk_embs, prot_mask)           # [B, D]

        # 3. Concatenate and compute logits
        rep    = torch.cat([d_rep, p_rep], dim=-1)               # [B, 2D]
        logits = self.gate_mlp(rep)                               # [B, num_experts]
        return logits


# ---------------------------------------------------------------------------
# Final Aggregation MLP
# ---------------------------------------------------------------------------
class FinalAggMLP(nn.Module):
    """
    Aggregates scalar predictions from all experts into a single output.

    Input  : [B, num_experts]  – weighted expert scalars (0 for non-selected)
    Output : [B, 1]

    This fully-connected MLP is trained end-to-end; its weights are updated
    through back-propagation just like every other module.
    """

    def __init__(self, num_experts: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_experts, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, 1]


# ---------------------------------------------------------------------------
# DTI Sparse MoE
# ---------------------------------------------------------------------------
class DTI_Sparse_MoE(nn.Module):
    """
    Sparse Mixture-of-Experts DTI model.

    Forward pass:
      1. Gate computes routing logits from token-level drug/protein embeddings.
      2. Top-k (k=2) experts are selected per sample.
      3. Selected experts run independently on their own encoded inputs.
      4. Expert scalar outputs * routing weights → [B, num_experts] matrix.
      5. FinalAggMLP maps [B, num_experts] → [B] prediction.
      6. Auxiliary load-balancing loss (Switch-Transformer CV² style) is added.

    Args:
        experts_dict : OrderedDict[str, nn.Module] – exactly 6 experts
        drug_vocab   : SMILES character vocabulary size for gating encoder
        prot_vocab   : amino-acid vocabulary size for gating encoder
        k            : top-k experts activated per sample  (default 2)
        lambda_aux   : weight for auxiliary load-balancing loss
    """

    def __init__(
        self,
        experts_dict: dict,
        drug_vocab:   int   = 66,
        prot_vocab:   int   = 26,
        k:            int   = 2,
        lambda_aux:   float = 0.1,
    ):
        super().__init__()
        self.num_experts  = len(experts_dict)
        self.expert_keys  = list(experts_dict.keys())
        self.experts      = nn.ModuleDict(experts_dict)
        self.k            = k
        self.lambda_aux   = lambda_aux

        # Gating network  (uses token-level embeddings internally)
        self.gate = SharedGatingNetwork(
            drug_vocab, prot_vocab, num_experts=self.num_experts
        )

        # Trainable final aggregation MLP  (weights updated during training)
        self.agg_mlp = FinalAggMLP(num_experts=self.num_experts, hidden=64)

    # ------------------------------------------------------------------
    def forward(self, batch: dict):
        """
        Args:
            batch : dict containing all expert-specific inputs plus gating tokens.
                    Required keys: 'label', 'shared_drug', 'shared_prot'
                    Optional keys: 'shared_drug_mask', 'shared_prot_mask'
                    Expert-specific keys: see dispatch logic below.
        Returns:
            final_output : [B]       – scalar predictions
            aux_loss     : scalar    – weighted load-balancing loss
        """
        B      = batch['label'].shape[0]
        device = next(self.parameters()).device


        drug_tokens = batch['shared_drug'].to(device)        # [B, L_d]
        prot_tokens = batch['shared_prot'].to(device)        # [B, L_p]
        drug_mask   = batch.get('shared_drug_mask')
        prot_mask   = batch.get('shared_prot_mask')
        if drug_mask is not None:
            drug_mask = drug_mask.to(device)
        if prot_mask is not None:
            prot_mask = prot_mask.to(device)

        # ---- 1. Routing -----------------------------------------------
        logits      = self.gate(drug_tokens, prot_tokens, drug_mask, prot_mask)
        gate_probs  = F.softmax(logits, dim=-1)                      # [B, E]
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.k, dim=-1)  # [B, k]

        # Normalize selected weights to sum to 1
        w = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-6)     # [B, k]

        # FIX 6: Enabled expert routing prints (controlled by MOE_DEBUG_ROUTING env var)
        if os.environ.get("MOE_DEBUG_ROUTING", "0") == "1":
            for i in range(B):
                selected = [self.expert_keys[idx] for idx in top_k_indices[i].tolist()]
                print(f"Sample {i} -> {selected}")

        # ---- 2. Expert dispatch ---------------------------------------
        # Collect weighted scalar per expert slot: [B, num_experts]
        expert_preds = torch.zeros(B, self.num_experts, device=device)

        # Track empirical routing fractions for aux loss (not differentiable)
        routing_fractions = torch.zeros(self.num_experts, device=device)

        for exp_idx, expert_name in enumerate(self.expert_keys):
            # Which samples selected this expert?
            mask             = (top_k_indices == exp_idx).any(dim=-1)  # [B]
            selected_indices = mask.nonzero(as_tuple=True)[0]
            routing_fractions[exp_idx] = len(selected_indices) / B

            if len(selected_indices) == 0:
                continue  # Expert not used in this batch

            expert = self.experts[expert_name]

            run_indices = selected_indices

            if expert_name == 'dp':
                # DeepPurpose features are already padded by robust moe_collate_fn
                af  = batch['dp_af'][run_indices].to(device)
                bf  = batch['dp_bf'][run_indices].to(device)
                ag  = batch['dp_ag'][run_indices].to(device)
                bg  = batch['dp_bg'][run_indices].to(device)
                abn = batch['dp_abn'][run_indices].to(device)
                pro = batch['dp_pro'][run_indices].to(device)
                out = expert([af, bf, ag, bg, abn], pro).reshape(-1)

            # NOTE: 'cpi' branch removed — expert is registered as 'perceivercpi'.
            # Shared embeddings must NEVER be passed to experts (pipeline spec).

            elif expert_name == 'dcdti':
                com = batch['dcdti_com'][run_indices].to(device)
                pro = batch['dcdti_pro'][run_indices].long().to(device)
                
                # Safety checks for embedding indices (DCDTI vocab is 2500)
                assert pro.max() < 2500, f"DCDTI protein index {pro.max()} exceeds vocab size 2500"
                assert pro.min() >= 0, "DCDTI protein index below 0"
                
                out = expert(com, pro).reshape(-1)

            elif expert_name == 'dpdta':
                com = batch['dpdta_com'][run_indices].to(device)
                pro = batch['dpdta_pro'][run_indices].to(device)
                out = expert(com, pro).reshape(-1)

            elif expert_name == 'mdprd':
                com = batch['mdprd_com'][run_indices].to(device)
                pro = batch['mdprd_pro'][run_indices].to(device)
                out = expert(com, pro).reshape(-1)

            elif expert_name == 'gifdti':
                com      = batch['gifdti_com'][run_indices].to(device)
                pro      = batch['gifdti_pro'][run_indices].to(device)
                com_mask = (batch['gifdti_com_mask'][run_indices].to(device)
                            if 'gifdti_com_mask' in batch else None)
                pro_mask = (batch['gifdti_pro_mask'][run_indices].to(device)
                            if 'gifdti_pro_mask' in batch else None)
                out = expert(com, pro, com_mask, pro_mask).reshape(-1)

            elif expert_name == 'perceivercpi':
                # pcpi_graph is a list of [atoms, bonds, adj] from collate_fn
                selected_graphs = [batch['pcpi_graph'][i] for i in run_indices.tolist()]
                
                # Pad PCPI graph components
                pcpi_atoms, pcpi_bonds, pcpi_adj = self._pad_pcpi_batch(selected_graphs, device)
                
                pcpi_morgan   = batch['pcpi_morgan'][run_indices].to(device)
                pcpi_sequence = batch['pcpi_sequence'][run_indices].to(device)
                out = expert([pcpi_atoms, pcpi_bonds, pcpi_adj],
                             pcpi_morgan, pcpi_sequence).reshape(-1)

            else:
                raise RuntimeError(f"Unsupported expert '{expert_name}' in MoE dispatch.")

            # ---- Accumulate weighted prediction into expert slot -------
            # For each selected sample find the routing weight for this expert
            # top_k_indices[selected_indices] : [S, k]
            weight_mask = (top_k_indices[selected_indices] == exp_idx)  # [S, k]
            exp_weights = w[selected_indices][weight_mask]               # [S_matched]
            if exp_weights.numel() != out.numel():
                raise RuntimeError(
                    f"Routing weight mismatch for expert '{expert_name}': "
                    f"{exp_weights.numel()} weights vs {out.numel()} outputs"
                )

            # exp_weights aligns with `out` because each sample in selected_indices
            # appears exactly once per expert slot (any() deduplicates).
            expert_preds[selected_indices, exp_idx] = out * exp_weights

        # ---- 3. Final MLP aggregation (trainable) ---------------------
        # expert_preds : [B, num_experts]  zeros for non-selected experts
        final_output = self.agg_mlp(expert_preds).squeeze(-1)  # [B]

        # ---- 4. Auxiliary load-balancing loss -------------------------
        mean_gate_probs = gate_probs.mean(dim=0)               # [E]
        routing_assignments = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        routing_fractions = routing_assignments.sum(dim=(0, 1)) / (B * self.k)
        aux_loss = self.num_experts * torch.sum(routing_fractions * mean_gate_probs)
        return final_output, aux_loss * self.lambda_aux

    def _pad_pcpi_batch(self, graphs, device):
        """Helper to pad a batch of pcpi_graph (atoms, bonds, adj)."""
        B = len(graphs)
        max_n = max(g[0].shape[0] for g in graphs)
        
        # atoms: [B, N, 5]
        atoms = torch.zeros(B, max_n, graphs[0][0].shape[1], device=device)
        # bonds: [B, N, N, 3]
        bonds = torch.zeros(B, max_n, max_n, graphs[0][1].shape[2], device=device)
        # adj: [B, N, N]
        adj = torch.zeros(B, max_n, max_n, device=device)
        
        for i, (a, b, ad) in enumerate(graphs):
            n = a.shape[0]
            atoms[i, :n, :] = a
            bonds[i, :n, :n, :] = b
            adj[i, :n, :n] = ad
            
        return atoms, bonds, adj
