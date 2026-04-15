
import torch
import torch.nn as nn
import torch.nn.functional as F
class GlobalMaxPooling1D(nn.Module):
    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
            pooled = torch.max(x, dim=1).values
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
            return pooled
        return torch.max(x, dim=1).values



class DeepDTA(nn.Module):
  def __init__(self, com_in_dim, pro_in_dim, out_dim, task):
      super().__init__()

      self.com_emb = nn.Embedding(com_in_dim, 128, padding_idx=0)    # com_in_dim = 85 (davis), 100 (kiba)
      self.com_layer = nn.Sequential(
        nn.Conv1d(128, 32, 4, stride=1, padding=0),
        nn.Conv1d(32, 32*2, 6, stride=1, padding=0),
        nn.Conv1d(32*2, 32*3, 8, stride=1, padding=0),
      )
      self.com_max_pool = GlobalMaxPooling1D()

      self.pro_emb = nn.Embedding(pro_in_dim, 128, padding_idx=0)    # pro_in_dim = 1200 (davis), 1000 (kiba)
      self.pro_layer = nn.Sequential(
        nn.Conv1d(128, 32, 4, stride=1, padding=0),
        nn.Conv1d(32, 32*2, 8, stride=1, padding=0),
        nn.Conv1d(32*2, 32*3, 12, stride=1, padding=0), 
      )
      self.pro_max_pool = GlobalMaxPooling1D()
      self.task = task
      
      self.int_layer = nn.Sequential(
        nn.Linear(192, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.1),
      )
      self.final = nn.Linear(512, out_dim)
      nn.init.normal_(self.final.weight)

  def _reduced_mask(self, tokens, kernels, out_len):
    valid_len = tokens.ne(0).sum(dim=1)
    reduction = sum(k - 1 for k in kernels)
    valid_out_len = (valid_len - reduction).clamp(min=1, max=out_len)
    positions = torch.arange(out_len, device=tokens.device).unsqueeze(0)
    return positions >= valid_out_len.unsqueeze(1)

  def forward(self, com, pro):
    # print(com.shape, pro.shape)
    com_tokens = com
    pro_tokens = pro

    com = self.com_emb(com)
    com = com.permute(0,2,1)

    pro = self.pro_emb(pro)
    pro = pro.permute(0,2,1)

    pro = self.pro_layer(pro).permute(0,2,1)
    com = self.com_layer(com).permute(0,2,1)

    pro = self.pro_max_pool(pro, self._reduced_mask(pro_tokens, [4, 8, 12], pro.size(1)))
    com = self.com_max_pool(com, self._reduced_mask(com_tokens, [4, 6, 8], com.size(1)))


    x = torch.cat((com, pro), 1)
    x = self.int_layer(x)
    x = self.final(x)
    # FIX 1: Output raw logits; BCEWithLogitsLoss handles sigmoid externally.
    return x
