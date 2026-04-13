import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Shared Embeddings built natively for Gateway
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [batch, seq_len, d_model]
        x = x + self.pe[:x.size(1), :]
        return x

class SharedGatingEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=2000, is_chunked=False, chunk_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.is_chunked = is_chunked
        self.chunk_size = chunk_size
        
        if is_chunked:
            self.chunk_attn = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # x: [B, L] token IDs. mask: [B, L] boolean True if padding.
        if self.is_chunked and x.size(1) > self.chunk_size:
            # Chunking logic for long proteins
            B, L = x.shape
            num_chunks = math.ceil(L / self.chunk_size)
            chunk_reps = []
            
            for i in range(num_chunks):
                start = i * self.chunk_size
                end = min((i + 1) * self.chunk_size, L)
                
                x_chunk = x[:, start:end]
                mask_chunk = mask[:, start:end] if mask is not None else None
                
                emb = self.embedding(x_chunk)
                emb = self.pos_encoder(emb)
                
                out = self.transformer(emb, src_key_padding_mask=mask_chunk)
                # Mean pool over sequence length for this chunk
                valid_mask = ~mask_chunk if mask_chunk is not None else torch.ones_like(x_chunk, dtype=torch.bool)
                # handle all pad chunks
                sum_out = (out * valid_mask.unsqueeze(-1)).sum(dim=1)
                valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
                chunk_rep = sum_out / valid_counts
                chunk_reps.append(chunk_rep)
                
            chunk_reps = torch.stack(chunk_reps, dim=1) # [B, num_chunks, D]
            
            # Attention across chunks
            attn_weights = torch.softmax(self.chunk_attn(chunk_reps), dim=1) # [B, num_chunks, 1]
            final_rep = (chunk_reps * attn_weights).sum(dim=1) # [B, D]
            return final_rep
            
        else:
            # Standard path
            emb = self.embedding(x)
            emb = self.pos_encoder(emb)
            out = self.transformer(emb, src_key_padding_mask=mask)
            # Pool
            if mask is not None:
                valid_mask = ~mask
                sum_out = (out * valid_mask.unsqueeze(-1)).sum(dim=1)
                valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
                return sum_out / valid_counts
            else:
                return out.mean(dim=1)

class SharedGatingNetwork(nn.Module):
    def __init__(self, drug_vocab, prot_vocab, d_model=128, num_experts=6):
        super().__init__()
        self.drug_enc = SharedGatingEncoder(drug_vocab, d_model, is_chunked=False)
        self.prot_enc = SharedGatingEncoder(prot_vocab, d_model, is_chunked=True, chunk_size=512)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_experts)
        )

    def forward(self, drug_tokens, prot_tokens, drug_mask=None, prot_mask=None):
        d_rep = self.drug_enc(drug_tokens, drug_mask)
        p_rep = self.prot_enc(prot_tokens, prot_mask)
        rep = torch.cat([d_rep, p_rep], dim=-1)
        logits = self.mlp(rep)
        return logits

class DTI_Sparse_MoE(nn.Module):
    def __init__(self, experts_dict, drug_vocab=65, prot_vocab=26, k=2, lambda_aux=0.1):
        super().__init__()
        self.num_experts = len(experts_dict)
        self.expert_keys = list(experts_dict.keys())
        self.experts = nn.ModuleDict(experts_dict)
        self.k = k
        self.lambda_aux = lambda_aux
        
        self.gate = SharedGatingNetwork(drug_vocab, prot_vocab, num_experts=self.num_experts)

    def forward(self, batch):
        """
        batch: unified dictionary containing all inputs needed for all experts,
               along with the gating tokens 'shared_drug' and 'shared_prot'.
        """
        B = batch['label'].shape[0]
        device = batch['label'].device
        
        drug_tokens = batch['shared_drug'].to(device)
        prot_tokens = batch['shared_prot'].to(device)
        drug_mask = batch['shared_drug_mask'].to(device) if 'shared_drug_mask' in batch else None
        prot_mask = batch['shared_prot_mask'].to(device) if 'shared_prot_mask' in batch else None
        
        # 1. Gate calculation
        logits = self.gate(drug_tokens, prot_tokens, drug_mask, prot_mask)
        gate_probs = F.softmax(logits, dim=-1) # [B, 6]
        
        # 2. Top-k routing
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.k, dim=-1) # [B, 2]
        
        # Normalize top-k probabilities to sum to 1
        w = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-6) # [B, 2]
        
        # 3. Batched Dispatching
        final_output = torch.zeros(B, device=device)
        
        # Track fractions for load balancing loss
        routing_fractions = torch.zeros(self.num_experts, device=device)
        
        # Only run loop over experts, but actively bypass using masks
        for exp_idx, expert_name in enumerate(self.expert_keys):
            # Find which samples in batch selected this expert
            mask = (top_k_indices == exp_idx).any(dim=-1) # [B]
            selected_indices = mask.nonzero(as_tuple=True)[0]
            
            # Record routing fraction
            routing_fractions[exp_idx] = len(selected_indices) / B
            
            if len(selected_indices) == 0:
                continue # Bypass completely!
                
            # Filter batch precisely for this expert to ensure GPU efficiency
            expert = self.experts[expert_name]
            
            # Call expert specific wrapper or extract features
            # (We implement this dynamically based on expert_name)
            if expert_name == 'dp':
                af = batch['dp_af'][selected_indices].to(device)
                bf = batch['dp_bf'][selected_indices].to(device)
                ag = batch['dp_ag'][selected_indices].to(device)
                bg = batch['dp_bg'][selected_indices].to(device)
                abn = batch['dp_abn'][selected_indices].to(device)
                pro = batch['dp_pro'][selected_indices].to(device)
                
                out = expert([af, bf, ag, bg, abn], pro).reshape(-1)
                
            elif expert_name == 'cpi':
                com = batch['cpi_com'][selected_indices].to(device)
                adj = batch['cpi_adj'][selected_indices].to(device)
                pro = batch['cpi_pro'][selected_indices].to(device)
                
                out = expert(com, adj, pro).reshape(-1)
                
            elif expert_name == 'dcdti':
                com = batch['dcdti_com'][selected_indices].to(device)
                pro = batch['dcdti_pro'][selected_indices].to(device)
                out = expert(com, pro).reshape(-1)
                
            elif expert_name == 'dpdta':
                com = batch['dpdta_com'][selected_indices].to(device)
                pro = batch['dpdta_pro'][selected_indices].to(device)
                out = expert(com, pro).reshape(-1)
                
            elif expert_name == 'mdprd':
                com = batch['mdprd_com'][selected_indices].to(device)
                pro = batch['mdprd_pro'][selected_indices].to(device)
                out = expert(com, pro).reshape(-1)
                
            elif expert_name == 'gifdti':
                # Simplified representation: wrap inputs if gifdti implemented
                com = batch['gifdti_com'][selected_indices].to(device)
                pro = batch['gifdti_pro'][selected_indices].to(device)
                out = expert(com, pro).reshape(-1)
                
            elif expert_name == 'perceivercpi':
                com = batch['shared_drug'][selected_indices].to(device)
                pro = batch['shared_prot'][selected_indices].to(device)
                com_mask = batch['shared_drug_mask'][selected_indices].to(device) if 'shared_drug_mask' in batch else None
                pro_mask = batch['shared_prot_mask'][selected_indices].to(device) if 'shared_prot_mask' in batch else None
                
                # Assuming perceiver uses integer tokens for embedding layers
                # Expand dims slightly to mimic expected shape
                out = expert(com.unsqueeze(-1), pro.unsqueeze(-1), com_mask, pro_mask).reshape(-1)
                
            else:
                out = torch.zeros(len(selected_indices), device=device) # Fallback
            
            # Combine the scattered outputs back using the computed weights
            # Find WHERE in top_k_indices this expert was located (0 or 1 index)
            # to multiply by the correct weight out of the top 2
            
            # boolean matrix mapping [B, 2] indicating positions of current expert
            weight_mask = (top_k_indices[selected_indices] == exp_idx)
            
            # extract corresponding weights
            exp_weights = w[selected_indices][weight_mask] 
            
            final_output[selected_indices] += out * exp_weights

        # 4. Auxiliary Load Balancing Loss
        # CV^2 of routing fractions.
        # Var(f) / Mean(f)^2 = sum(f_i * P_i) * num_experts (approx for continuous differentiability)
        mean_gate_probs = gate_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(routing_fractions * mean_gate_probs)
        
        return final_output, aux_loss * self.lambda_aux
