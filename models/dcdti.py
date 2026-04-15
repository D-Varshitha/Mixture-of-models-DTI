
import torch
import torch.nn as nn
class GlobalMaxPooling1D(nn.Module):
    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
            pooled = torch.max(x, dim=1).values
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
            return pooled
        return torch.max(x, dim=1).values

class DeepConvDTI(nn.Module):
  ''' 
      train with l2 loss decay=0.001
      initialize model parameters with glorot normalization
  '''

  def __init__(self, out_dim, task):
    super().__init__()
    self.pro_layer_init = nn.Sequential(
      nn.Embedding(2500, 20, padding_idx=0),
      nn.Dropout1d(0.2),
    )
    self.pro_layer_cnn = nn.ModuleList(
      [
        nn.Sequential(
          nn.Conv1d(20, 128, ch, stride=1, padding=0),
          nn.BatchNorm1d(128),
          nn.ReLU(),
        )
        for ch in [10,15,20,25,30]
      ]
    )
    self.max_pool = GlobalMaxPooling1D()
    self.pro_layer = nn.Sequential(
      nn.Linear(640, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Dropout(0.2),
    )
    self.com_layer = nn.Sequential(
      nn.Linear(2048, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(512, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Dropout(0.2),
    )
    self.int_layer = nn.Sequential(
      nn.Linear(256, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Linear(128, out_dim)
    )

    for m in self.modules():
      if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)

    self.task = task

  def _branch_mask(self, tokens, kernel_size, out_len):
    valid_len = tokens.ne(0).sum(dim=1)
    valid_out_len = (valid_len - (kernel_size - 1)).clamp(min=1, max=out_len)
    positions = torch.arange(out_len, device=tokens.device).unsqueeze(0)
    return positions >= valid_out_len.unsqueeze(1)

    
  def forward(self, com, pro):
    pro_tokens = pro

    pro = self.pro_layer_init(pro)

    pro = pro.permute(0,2,1)

    pro_1 = self.pro_layer_cnn[0](pro).permute(0,2,1)
    pro_2 = self.pro_layer_cnn[1](pro).permute(0,2,1)
    pro_3 = self.pro_layer_cnn[2](pro).permute(0,2,1)
    pro_4 = self.pro_layer_cnn[3](pro).permute(0,2,1)
    pro_5 = self.pro_layer_cnn[4](pro).permute(0,2,1)

    pro_1 = self.max_pool(pro_1, self._branch_mask(pro_tokens, 10, pro_1.size(1)))
    pro_2 = self.max_pool(pro_2, self._branch_mask(pro_tokens, 15, pro_2.size(1)))
    pro_3 = self.max_pool(pro_3, self._branch_mask(pro_tokens, 20, pro_3.size(1)))
    pro_4 = self.max_pool(pro_4, self._branch_mask(pro_tokens, 25, pro_4.size(1)))
    pro_5 = self.max_pool(pro_5, self._branch_mask(pro_tokens, 30, pro_5.size(1)))

    pro = torch.cat((pro_1, pro_2, pro_3, pro_4, pro_5), 1)

    pro = self.pro_layer(pro)
    com = self.com_layer(com)

    x = torch.cat((com, pro), 1)
    x = self.int_layer(x)
    # FIX 1: Output raw logits; BCEWithLogitsLoss handles sigmoid externally.
    return x
