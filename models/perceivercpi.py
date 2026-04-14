"""
PerceiverCPI Expert Model
=========================
Based on: https://github.com/dmis-lab/PerceiverCPI
Paper: "PerceiverCPI: A nested cross-attention network for compound-protein
         interaction prediction" (Bioinformatics 2022)

Architecture matched to the original paper:
  - Compound: D-MPNN (graph) + Morgan fingerprint (ECFP) → Cross-Attention with protein
  - Protein : Embedding → 1D-CNN stack (GLU gates, residual) → flattened
  - CrossAttentionBlock (CAB):
       (1) Cross-attn: enrich graph feature using Morgan feature
       (2) Self-attn on graph feature
       (3) Cross-attn: graph feature queries protein feature
  - Final FFN for prediction

In the MoE context this expert receives *integer token-ID sequences* for both
the compound (SMILES chars) and protein (amino-acid chars).  We therefore use
learnable embeddings internally – no external pretrained embeddings are required
for this expert's own forward pass.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention block (single-head, matches original CAB implementation)
# ---------------------------------------------------------------------------
class AttentionBlock(nn.Module):
    """Scaled dot-product attention with projection (Q, K, V each linear)."""

    def __init__(self, hid_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)
        self.fc  = nn.Linear(hid_dim, hid_dim)
        self.do  = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query, key, value: [B, hid_dim] – pooled 1-D representations.
        Returns:
            [B, hid_dim]
        """
        B = query.shape[0]

        Q = self.f_q(query)  # [B, D]
        K = self.f_k(key)
        V = self.f_v(value)

        # Reshape to [B, n_heads, 1, head_dim] for batched matmul
        Q = Q.view(B, self.n_heads, self.head_dim).unsqueeze(2)   # [B, H, 1, D/H]
        K = K.view(B, self.n_heads, self.head_dim).unsqueeze(2)
        V = V.view(B, self.n_heads, self.head_dim).unsqueeze(2)

        energy  = torch.matmul(Q, K.transpose(2, 3)) / self.scale # [B, H, 1, 1]
        attn    = self.do(F.softmax(energy, dim=-1))               # [B, H, 1, 1]
        out     = torch.matmul(attn, V)                            # [B, H, 1, D/H]

        out = out.squeeze(2).contiguous().view(B, self.hid_dim)    # [B, D]
        return self.do(self.fc(out))


# ---------------------------------------------------------------------------
# Cross-Attention Block  (CAB) – mirrors the original repo exactly
# ---------------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """
    Three-step interaction block from PerceiverCPI:
        step 1: cross-attn – enrich graph_feat with morgan_feat
        step 2: self-attn  – refine graph_feat
        step 3: cross-attn – graph_feat queries protein_feat → output
    """

    def __init__(self, hid_dim: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.att = AttentionBlock(hid_dim, n_heads, dropout)

    def forward(self, graph_feat: torch.Tensor, morgan_feat: torch.Tensor,
                sequence_feat: torch.Tensor) -> torch.Tensor:
        # step 1: cross-attn  (morgan enriches graph)
        graph_feat = graph_feat + self.att(morgan_feat, graph_feat, graph_feat)
        # step 2: self-attn   (refine)
        graph_feat = self.att(graph_feat, graph_feat, graph_feat)
        # step 3: cross-attn  (graph queries protein)
        output = self.att(graph_feat, sequence_feat, sequence_feat)
        return output


# ---------------------------------------------------------------------------
# Dense MPNN
# ---------------------------------------------------------------------------
class DenseMPNN(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, hidden_dim: int):
        super().__init__()
        self.W_v = nn.Linear(atom_dim, hidden_dim)
        self.W_e = nn.Linear(bond_dim, hidden_dim)
        self.W_msg = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_upd = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, atoms: torch.Tensor, bonds: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # atoms: [B, N, atom_dim]
        # bonds: [B, N, N, bond_dim]
        # adj:   [B, N, N]
        H = self.W_v(atoms) # [B, N, hidden]
        E = self.W_e(bonds) # [B, N, N, hidden]

        B, N, _ = H.shape

        for _ in range(3):
            # Form messages using neighboring node features
            H_j = H.unsqueeze(1).expand(B, N, N, -1)
            msg = self.W_msg(torch.cat([H_j, E], dim=-1))
            msg = F.relu(msg)
            
            # Mask out non-adjacent messages
            msg = msg * adj.unsqueeze(-1)
            
            # Aggregate neighbors
            agg_msg = msg.sum(dim=2) # [B, N, hidden]
            
            # Update node features
            upd = self.W_upd(torch.cat([H, agg_msg], dim=-1))
            H = F.relu(upd)
            
        # Valid tokens have sum of absolute atom features > 0
        mask = (atoms.abs().sum(-1) > 0).float() # [B, N]
        H = H * mask.unsqueeze(-1)
        
        # Graph-level feature (mean pool over valid atoms)
        graph_feat = H.sum(dim=1) / mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return graph_feat



# ---------------------------------------------------------------------------
# PerceiverCPI Expert
# ---------------------------------------------------------------------------
class PerceiverCPI(nn.Module):
    """
    Standalone PerceiverCPI expert for use inside the DTI-MoE system.

    Inputs (from MoEDataset / model_MoE.py):
        drug_seq  : [B, L_d]  – integer SMILES token IDs  (shared_drug)
        prot_seq  : [B, L_p]  – integer amino-acid IDs    (shared_prot)
        drug_mask : [B, L_d]  – True where padding        (shared_drug_mask)
        prot_mask : [B, L_p]  – True where padding        (shared_prot_mask)

    Internal pipeline:
        drug_seq  → Embedding → 1-D CNN → global-max-pool  → graph_feat [B, hidden]
        drug_seq  → Embedding → mean-pool (simulates Morgan) → morgan_feat [B, hidden]
        prot_seq  → Embedding → 1-D CNN (GLU, residual)    → flat → FC → seq_feat [B, hidden]
        CAB(graph_feat, morgan_feat, seq_feat)              → interaction [B, hidden]
        FFN → scalar prediction
    """

    def __init__(
        self,
        drug_vocab:   int   = 66,    # SMILES char vocab size (+1 for padding)
        prot_vocab:   int   = 26,    # amino-acid vocab size  (+1 for padding)
        hidden_size:  int   = 128,   # shared hidden dimension (hid_dim in CAB)
        prot_seq_len: int   = 1000,  # max protein sequence length (for FC sizing)
        prot_cnn_out: int   = 32,    # number of 1-D CNN output channels for protein
        num_cnn:      int   = 3,     # number of CNN layers for protein
        kernel_size:  int   = 7,     # kernel size for protein CNN
        dropout:      float = 0.1,
        output_dim:   int   = 1,
        task:         str   = 'classification',
    ):
        super().__init__()
        self.task = task

        # ---- compound: D-MPNN for graph, and proj for Morgan ----
        self.mpnn = DenseMPNN(atom_dim=5, bond_dim=3, hidden_dim=hidden_size)
        self.morgan_proj = nn.Linear(1024, hidden_size)

        # ---- protein: embedding + stacked GLU-CNN (matches original) ---------
        self.prot_emb        = nn.Embedding(prot_vocab, hidden_size, padding_idx=0)
        self.prot_conv_in    = nn.Conv1d(prot_seq_len, prot_cnn_out, kernel_size=1)
        self.prot_convs      = nn.ModuleList([
            nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size, padding=kernel_size // 2)
            for _ in range(num_cnn)
        ])
        self.prot_residual   = nn.Linear(hidden_size, prot_cnn_out)
        self.prot_norm       = nn.LayerNorm(prot_cnn_out)
        self.prot_fc         = nn.Linear(hidden_size * prot_cnn_out, hidden_size)

        # ---- cross-attention block -------------------------------------------
        self.cab = CrossAttentionBlock(hidden_size, n_heads=1, dropout=dropout)

        # ---- final FFN -------------------------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_dim),
        )

        if task == 'classification':
            self.sigmoid = nn.Sigmoid()

    # ------------------------------------------------------------------
    def _encode_prot(self, prot_seq: torch.Tensor,
                     prot_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Returns sequence_feat [B, hidden_size].
        Mirrors the original PerceiverCPI protein 1-D CNN pipeline.
        """
        emb = self.prot_emb(prot_seq)           # [B, L, H]

        # conv_in maps [B, L, H] → [B, prot_cnn_out, H]  (sequence dim is treated as channels)
        conv_input = self.prot_conv_in(emb)      # [B, prot_cnn_out, H]
        conv_input = conv_input.permute(0, 2, 1) # [B, H, prot_cnn_out]

        for conv in self.prot_convs:
            # conv expects [B, H, prot_cnn_out] → outputs [B, 2H, prot_cnn_out]
            conved = conv(conv_input)            # [B, 2H, prot_cnn_out]
            conved = F.glu(conved, dim=1)        # [B, H, prot_cnn_out]   (GLU halves channels)
            # residual: project conv_input to match if dims differ
            conv_input = conved + conv_input     # [B, H, prot_cnn_out]

        # Flatten + FC
        B = prot_seq.shape[0]
        flat = conv_input.contiguous().view(B, -1)          # [B, H * prot_cnn_out]
        seq_feat = self.dropout(F.relu(self.prot_fc(flat))) # [B, H]
        return seq_feat

    def forward(
        self,
        pcpi_graph:    list,
        pcpi_morgan:   torch.Tensor,
        pcpi_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pcpi_graph   : Tuple of [atoms, bonds, adj] tensors
            pcpi_morgan  : [B, 1024] Morgan FP
            pcpi_sequence: [B, L_p] long – integer amino-acid IDs
        Returns:
            [B, 1] predictions (sigmoid applied for classification)
        """
        atoms, bonds, adj = pcpi_graph
        graph_feat = self.mpnn(atoms, bonds, adj)          # [B, H]
        morgan_feat = self.morgan_proj(pcpi_morgan)        # [B, H]
        seq_feat = self._encode_prot(pcpi_sequence)        # [B, H]

        interaction = self.cab(graph_feat, morgan_feat, seq_feat)   # [B, H]
        out         = self.ffn(interaction)                          # [B, 1]

        if self.task == 'classification' and not self.training:
            out = self.sigmoid(out)

        return out
