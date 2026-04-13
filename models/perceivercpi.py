import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PerceiverBlock(nn.Module):
    def __init__(self, latent_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_dim)
        self.self_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self attention over latents
        x_ln = self.ln1(x)
        attn_out, _ = self.self_attn(x_ln, x_ln, x_ln)
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.ln2(x))
        return x

class CrossAttention(nn.Module):
    def __init__(self, latent_dim, context_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln_latents = nn.LayerNorm(latent_dim)
        self.ln_context = nn.LayerNorm(context_dim)
        
        # We need query matching Context dimension, or map context to latent
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(context_dim, latent_dim)
        self.v_proj = nn.Linear(context_dim, latent_dim)
        
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(self, latents, context, context_mask=None):
        lat = self.ln_latents(latents)
        ctx = self.ln_context(context)
        
        q = self.q_proj(lat)
        k = self.k_proj(ctx)
        v = self.v_proj(ctx)
        
        # Context mask shape: (batch, context_len) where True means ignore.
        attn_out, _ = self.cross_attn(query=q, key=k, value=v, key_padding_mask=context_mask)
        latents = latents + attn_out
        latents = latents + self.ffn(self.ln_latents(latents))
        return latents

class PerceiverCPI(nn.Module):
    def __init__(self, num_latents=32, latent_dim=128, context_dim=128, num_heads=4, depth=2, output_dim=1, task='classification'):
        super().__init__()
        self.task = task
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        
        # Learnable latents [num_latents, latent_dim]
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        
        # Project inputs to context dim if necessary
        self.drug_proj = nn.Sequential(nn.Linear(39, context_dim), nn.GELU()) # DeepPurpose atom features etc. or SMILES embeddings
        self.prot_proj = nn.Sequential(nn.Linear(26, context_dim), nn.GELU()) # Amino acids
        
        self.cross_attn = CrossAttention(latent_dim, context_dim, num_heads)
        self.self_attn_blocks = nn.ModuleList([
            PerceiverBlock(latent_dim, num_heads) for _ in range(depth)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, drug_seq, prot_seq, drug_mask=None, prot_mask=None):
        """
        drug_seq: [B, L_d, F_d] or indices
        prot_seq: [B, L_p, F_p] or indices
        """
        B = drug_seq.shape[0]
        
        # Map to context
        if drug_seq.dim() == 2: # IDs
            # Simple embedding fallback
            pass # Usually handled upstream
            
        d_ctx = self.drug_proj(drug_seq.float()) if drug_seq.dim() == 3 else drug_seq
        p_ctx = self.prot_proj(prot_seq.float()) if prot_seq.dim() == 3 else prot_seq
        
        # Combine contexts
        context = torch.cat([d_ctx, p_ctx], dim=1) # [B, L_d + L_p, C]
        
        # Combine masks
        if drug_mask is not None and prot_mask is not None:
            ctx_mask = torch.cat([drug_mask, prot_mask], dim=1)
        else:
            ctx_mask = None
            
        latents = self.latents.expand(B, -1, -1)
        
        # Cross-attend context into latents
        latents = self.cross_attn(latents, context, ctx_mask)
        
        # Process latents
        for block in self.self_attn_blocks:
            latents = block(latents)
            
        # Global average pooling on latents
        out = latents.mean(dim=1)
        
        out = self.output_layer(out)
        if self.task == 'classification' and out.shape[-1] == 1:
            out = torch.sigmoid(out)
            
        return out
