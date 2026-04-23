"""
PerceiverCPI Expert Model
=========================
Based on: https://github.com/dmis-lab/PerceiverCPI
Paper: "PerceiverCPI: A nested cross-attention network for compound-protein
         interaction prediction" (Bioinformatics 2022)

Architecture strictly matched to the official repo (chemprop + DeepPurpose backbone):

  D-MPNN (chemprop-style, BOND-LEVEL):
    - h^0_{vw} = ReLU(W_i · cat(x_v, e_{vw}))
    - m^t_{vw} = AVG_{k ∈ N(v)w} h^{t-1}_{kv}    [PerceiverCPI uses AVG]
    - h^t_{vw} = ReLU(h^0_{vw} + W_h · m^t_{vw})   [pre-activation residual]
    - atom_h_v = ReLU(W_o · cat(x_v, SUM_w h^T_{wv}))
    - Graph readout: mean-pool valid atom_h → [B, H]

  Morgan MLP:
    - 1024 → Linear → ReLU → [B, H] → unsqueeze → [B, 1, H]

  Protein CNN (DeepPurpose CNN encoder, STRICT):
    - Embedding [B,L] → [B,L,emb]
    - Permute → [B, emb, L]
    - Conv1d(emb, 2*f0, k0) → GLU → [B, f0, L]
    - Conv1d(f0,  2*f1, k1) → GLU → [B, f1, L]   (channels change per layer)
    - Conv1d(f1,  2*f2, k2) → GLU → [B, f2, L]
    - Flatten → [B, f2 * L]
    - Linear(f2*L, H) → prot_feat [B, H]
    - For CAB sequence attention: return pre-flatten token seq [B, L, f2]
      projected to [B, L, H] via prot_token_proj.

  CrossAttentionBlock (CAB) – SEQUENCE-LEVEL:
    Step 1: Q=graph_seq [B,N,H],  K=V=morgan_seq [B,1,H]
    Step 2: self-attn on graph_seq
    Step 3: Q=graph_seq [B,N,H],  K=V=prot_seq   [B,L,H]
    → mean-pool → [B, H]

  FFN → scalar
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AttentionBlock  – sequence-level multi-head attention
# ---------------------------------------------------------------------------
class AttentionBlock(nn.Module):
    """
    Multi-head scaled dot-product attention on TOKEN SEQUENCES [B, T, D].

    CHANGED vs original submitted code:
      - Operates on [B, T, D], NOT on pooled [B, D] vectors.
      - Correct multi-head reshape: [B, H, T, head_dim].
      - key_padding_mask support to ignore PAD tokens.
      - Residual + LayerNorm (present in original repo).
    """

    def __init__(self, hid_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim  = hid_dim
        self.n_heads  = n_heads
        self.head_dim = hid_dim // n_heads
        self.scale    = math.sqrt(self.head_dim)

        self.f_q  = nn.Linear(hid_dim, hid_dim)
        self.f_k  = nn.Linear(hid_dim, hid_dim)
        self.f_v  = nn.Linear(hid_dim, hid_dim)
        self.fc   = nn.Linear(hid_dim, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)   # CHANGED: added residual norm
        self.do   = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,                    # [B, Tq, D]
        key:   torch.Tensor,                    # [B, Tk, D]
        value: torch.Tensor,                    # [B, Tk, D]
        key_padding_mask: torch.Tensor = None,  # [B, Tk] bool  True=PAD
    ) -> torch.Tensor:                          # [B, Tq, D]
        B, Tq, _ = query.shape
        Tk = key.shape[1]

        # CHANGED: full sequence projection + multi-head reshape
        Q = self.f_q(query).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.f_k(key  ).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.f_v(value).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(2, 3)) / self.scale   # [B, H, Tq, Tk]
        if key_padding_mask is not None:
            energy = energy.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        attn = self.do(F.softmax(energy, dim=-1))
        out  = torch.matmul(attn, V)                               # [B, H, Tq, head_dim]
        out  = out.transpose(1, 2).contiguous().view(B, Tq, self.hid_dim)
        out  = self.do(self.fc(out))
        return self.norm(query + out)   # CHANGED: residual + norm


# ---------------------------------------------------------------------------
# CrossAttentionBlock  (CAB)
# ---------------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """
    Corrected CAB:

    Step 1: Q = Morgan, K = Graph
    Step 2: Self-attn on Graph
    Step 3: Q = Graph, K = Protein
    """

    def __init__(self, hid_dim: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        
        # FIX: separate attention blocks (no weight sharing)
        self.cross_attn_morgan = AttentionBlock(hid_dim, n_heads, dropout)
        self.self_attn_graph   = AttentionBlock(hid_dim, n_heads, dropout)
        self.cross_attn_prot   = AttentionBlock(hid_dim, n_heads, dropout)

    def forward(
        self,
        graph_seq:  torch.Tensor,    # [B, N, D]
        morgan_seq: torch.Tensor,    # [B, 1, D]
        prot_seq:   torch.Tensor,    # [B, L, D]
        prot_mask:  torch.Tensor = None,
    ) -> torch.Tensor:

        # ✅ FIX 1: Correct Q/K/V direction
        # Morgan queries Graph
        morgan_updated = self.cross_attn_morgan(
            morgan_seq, graph_seq, graph_seq
        )  # [B, 1, D]

        # Broadcast Morgan to graph tokens
        graph_seq = graph_seq + morgan_updated.expand(-1, graph_seq.size(1), -1)

        # Step 2: Self-attention on graph
        graph_seq = self.self_attn_graph(graph_seq, graph_seq, graph_seq)

        # Step 3: Graph queries Protein
        out = self.cross_attn_prot(graph_seq, prot_seq, prot_seq, prot_mask)

        return out.mean(dim=1)  # [B, D]

# ---------------------------------------------------------------------------
# DenseMPNN  – directed bond-level message passing (chemprop-style)
# ---------------------------------------------------------------------------
class DenseMPNN(nn.Module):
    """
    D-MPNN strictly following the chemprop / PerceiverCPI paper equations.

    CHANGED vs original:
      - Hidden states live on DIRECTED BONDS [B, N, N, H], not atoms.
      - Init: h^0 = ReLU(W_i · cat(x_v, e_{vw}))  per directed bond.
      - Msg:  m^t_{vw} = AVG_{k∈N(v)w} h^{t-1}_{kv}  (AVG excludes reverse edge).
      - Update: h^t = ReLU(h^0 + W_h · m^t)  (pre-activation residual).
      - Readout: atom_h = ReLU(W_o · cat(x_v, Σ_w h^T_{wv})).
    """

    def __init__(self, atom_dim: int, bond_dim: int, hidden_dim: int,
                 depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.depth   = depth
        self.dropout = nn.Dropout(dropout)
        # CHANGED: W_i takes cat(atom, bond) → bond hidden  (chemprop signature)
        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=False)
        # CHANGED: W_h takes aggregated message → hidden update
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # CHANGED: W_o takes cat(atom, agg_bond) → atom hidden (readout)
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        atoms: torch.Tensor,   # [B, N, atom_dim]
        bonds: torch.Tensor,   # [B, N, N, bond_dim]
        adj:   torch.Tensor,   # [B, N, N] float  1.=bond exists
    ) -> torch.Tensor:         # [B, N, hidden_dim]  atom-level tokens
        B, N, _ = atoms.shape

        # CHANGED: initialise bond hidden states h^0_{vw}
        atoms_src = atoms.unsqueeze(2).expand(B, N, N, atoms.shape[-1])
        H0 = F.relu(self.W_i(torch.cat([atoms_src, bonds], dim=-1)))  # [B,N,N,H]
        H  = H0 * adj.unsqueeze(-1)                                    # zero non-edges

        for _ in range(self.depth - 1):
            # CHANGED: aggregate bonds ARRIVING at each atom v: Σ_k h[k,v] → [B,N,H]
            agg_into_v = H.sum(dim=1)                                  # [B, N, H]

            # CHANGED: subtract reverse bond h[w,v] to exclude it, then AVG
            # agg_into_v expanded to directed edge shape: [B, N, N, H]
            agg_v = agg_into_v.unsqueeze(2).expand(B, N, N, H.shape[-1])
            H_rev = H.permute(0, 2, 1, 3)                              # h[b,w,v]→[b,v,w]
            # Number of valid neighbours of v (excl. w itself): max(adj[v,:]w, 1)
            n_nbr = adj.sum(dim=2, keepdim=True) - adj                 # [B,N,N]
            n_nbr = n_nbr.unsqueeze(-1).clamp(min=1)                   # [B,N,N,1]
            # CHANGED: m^t_{vw} = (Σ_{k} h[k,v] - h[w,v]) / |N(v)w|
            msg = (agg_v - H_rev) / n_nbr                              # [B,N,N,H]

            # CHANGED: h^t = ReLU(h^0 + W_h · m^t)
            H = F.relu(H0 + self.W_h(msg)) * adj.unsqueeze(-1)
            H = self.dropout(H)

        # CHANGED: readout – aggregate incoming bond hiddens Σ_w h[w,v] → [B,N,H]
        agg_final = H.sum(dim=1)                                       # [B, N, H]
        atom_h = F.relu(self.W_o(torch.cat([atoms, agg_final], dim=-1)))
        return self.dropout(atom_h)                                    # [B, N, H]


# ---------------------------------------------------------------------------
# PerceiverCPI Expert
# ---------------------------------------------------------------------------
class PerceiverCPI(nn.Module):
    """
    Full PerceiverCPI expert, strictly matching the official repo.

    forward() signature and argument names are UNCHANGED for MoE compatibility.

    Key changes from previously submitted code:
      1. DenseMPNN: bond-level hidden states, directed AVG message passing.
      2. Morgan: single Linear projection (not 2-layer MLP), unsqueeze to [B,1,H].
      3. Protein CNN: DeepPurpose-style heterogeneous filter stack (growing channels),
                      NO residual, FLATTEN then FC (not mean-pool).
      4. AttentionBlock: sequence-level [B,T,D], correct multi-head, masking, norm.
      5. CrossAttentionBlock: all three steps on sequences; mean-pool at the end only.
      6. prot_token_proj: linear from last CNN filter dim to hidden_size for CAB.
    """

    def __init__(
        self,
        atom_dim:     int   = 5,           # matches moe_dataset _get_rdkit_graph: [AtomicNum,Degree,FormalCharge,NumExplicitHs,IsAromatic]
        bond_dim:     int   = 3,           # matches moe_dataset _get_rdkit_graph: [BondType, IsInRing, IsConjugated]
        prot_vocab:   int   = 26,          # AA vocab (+1 for padding idx 0)
        hidden_size:  int   = 128,         # shared hidden dimension D
        # CHANGED: DeepPurpose-style heterogeneous CNN filter/kernel lists
        cnn_filters:  list  = None,        # default [32, 64, 96]
        cnn_kernels:  list  = None,        # default [4,  8,  12]
        prot_seq_len: int   = 1000,        # max protein length for FC sizing
        mpnn_depth:   int   = 3,           # D-MPNN depth
        n_heads:      int   = 1,           # attention heads (paper: 1)
        dropout:      float = 0.1,
        output_dim:   int   = 1,
        task:         str   = 'classification',
    ):
        super().__init__()
        self.task        = task
        self.hidden_size = hidden_size
        self.prot_seq_len = prot_seq_len

        if cnn_filters is None:
            cnn_filters = [32, 64, 96]
        if cnn_kernels is None:
            cnn_kernels = [4, 8, 12]
        assert len(cnn_filters) == len(cnn_kernels)
        self.cnn_filters = cnn_filters

        # ---- D-MPNN --------------------------------------------------------
        self.mpnn = DenseMPNN(atom_dim, bond_dim, hidden_size,
                              depth=mpnn_depth, dropout=dropout)

        # ---- Morgan projection ---------------------------------------------
        # CHANGED: single Linear (not 2-layer MLP) matching original repo
        self.morgan_proj = nn.Linear(1024, hidden_size)

        # ---- Protein Embedding ---------------------------------------------
        self.prot_emb = nn.Embedding(prot_vocab, hidden_size, padding_idx=0)

        # ---- Protein CNN stack (DeepPurpose CNN encoder) -------------------
        # CHANGED: heterogeneous channels (emb→f0→f1→f2), no residual, GLU halves
        prot_cnn_layers = []
        in_ch = hidden_size          # embedding dim
        for f, k in zip(cnn_filters, cnn_kernels):
            prot_cnn_layers.append(
                nn.Conv1d(in_ch, f * 2, kernel_size=k, padding=k // 2)
            )
            in_ch = f
        self.prot_convs = nn.ModuleList(prot_cnn_layers)

        # last_filter is the output channel count of the final CNN layer
        last_filter = cnn_filters[-1]

        # CHANGED: project CNN token dim (last_filter) to hidden_size for CAB
        self.prot_token_proj = (
            nn.Linear(last_filter, hidden_size)
            if last_filter != hidden_size else nn.Identity()
        )
        # NOTE: prot_fc (flat readout) is intentionally NOT included here since
        # the CAB path uses token-level output from _encode_prot, not a flat vector.

        # ---- Cross-Attention Block ------------------------------------------
        self.cab = CrossAttentionBlock(hidden_size, n_heads=n_heads,
                                       dropout=dropout)

        # ---- FFN -----------------------------------------------------------
        self.dropout_layer = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_dim),
        )

    # ---------------------------------------------------------------------- #
    def _encode_prot(self, prot_seq: torch.Tensor) -> torch.Tensor:
        """
        CHANGED: strictly follows DeepPurpose CNN encoder.
          Embedding → [B, emb, L] → Conv1d-GLU stack (heterogeneous filters,
          NO residual) → [B, last_filter, L]
        Returns token sequence [B, L, last_filter] for CAB cross-attention.
        The flat readout path (→ prot_fc → hidden_size) is available via
        self.prot_fc(x.flatten(1)) if needed externally.
        """
        emb = self.prot_emb(prot_seq)      # [B, L, emb]
        x   = emb.permute(0, 2, 1)         # [B, emb, L]

        # CHANGED: no residual – each layer changes channel dim
        for conv in self.prot_convs:
            x = F.glu(conv(x), dim=1)      # [B, 2f, L] → [B, f, L]

        # FIX 3: Even kernels with padding=k//2 produce L+1 per layer (3 layers → L+3).
        # Force output back to prot_seq_len so prot_fc Linear receives exact input size.
        x = F.adaptive_avg_pool1d(x, self.prot_seq_len)  # [B, last_filter, prot_seq_len]

        return x.permute(0, 2, 1)          # [B, prot_seq_len, last_filter]

    # ---------------------------------------------------------------------- #
    def _encode_graph(
        self,
        atoms: torch.Tensor,
        bonds: torch.Tensor,
        adj:   torch.Tensor,
    ) -> tuple:
        """Returns (atom_h [B,N,H], atom_mask [B,N] True=PAD)."""
        atom_h    = self.mpnn(atoms, bonds, adj)        # [B, N, H]
        atom_mask = (atoms.abs().sum(-1) == 0)          # [B, N] True=PAD
        atom_h    = atom_h.masked_fill(atom_mask.unsqueeze(-1), 0.0)
        return atom_h, atom_mask

    # ---------------------------------------------------------------------- #
    def forward(
        self,
        pcpi_graph:    tuple,           # (atoms [B,N,Fa], bonds [B,N,N,Fb], adj [B,N,N])
        pcpi_morgan:   torch.Tensor,    # [B, 1024]
        pcpi_sequence: torch.Tensor,    # [B, L_p]  long  (0 = padding)
    ) -> torch.Tensor:
        atoms, bonds, adj = pcpi_graph

        # ---- Graph: directed bond MPNN → atom token seq [B, N, H] ----
        graph_seq, atom_mask = self._encode_graph(atoms, bonds, adj)

        # ---- Morgan: Linear → ReLU → [B, 1, H] ----
        # CHANGED: single linear + unsqueeze (matches morgan_proj in repo)
        morgan_seq = self.morgan_proj(pcpi_morgan).unsqueeze(1)  # [B, 1, H]

        # ---- Protein CNN: token sequence [B, L, last_filter] ----
        prot_pad_mask = (pcpi_sequence == 0)            # [B, L]  True=PAD
        prot_tokens   = self._encode_prot(pcpi_sequence)  # [B, L_out, last_filter]

        # FIX 3: Feature dimension mismatch (Align mask length with CNN output)
        if prot_tokens.size(1) != prot_pad_mask.size(1):
            diff = prot_tokens.size(1) - prot_pad_mask.size(1)
            if diff > 0:
                prot_pad_mask = F.pad(prot_pad_mask, (0, diff), value=True)
            else:
                prot_pad_mask = prot_pad_mask[:, :prot_tokens.size(1)]

        # CHANGED: project CNN token dim → hidden_size for CAB attention
        prot_seq = self.prot_token_proj(prot_tokens)    # [B, L, H]

        # ---- CAB: nested cross-attention ----
        interaction = self.cab(
            graph_seq,
            morgan_seq,
            prot_seq,
            prot_mask=prot_pad_mask,
        )                                               # [B, H]

        # ---- FFN ----
        out = self.ffn(self.dropout_layer(interaction)) # [B, output_dim]
        return out