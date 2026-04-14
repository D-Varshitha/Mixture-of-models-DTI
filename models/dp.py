
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple




class DeepPurpose(nn.Module):
  def __init__(self, dims, in_ch, kernel, task):
    super().__init__()
    self.dropout = nn.Dropout(0.1)
    # hidden_dims = [1024, 1024, 512], input_dim_drug = 1024, input_dim_protein = 8420
    # dims = [input_dim_drug+input_dim_protein] + hidden_dims + [2]

    self.com_Wi = nn.Linear(39 + 11, 256, bias=False)
    self.com_Wh = nn.Linear(256, 256, bias=False)
    self.com_Wo = nn.Linear(39 + 256, 256)
    self.com_net_depth = 3

    self.pro_cnn = nn.ModuleList([nn.Conv1d(in_ch[i], in_ch[i+1], kernel[i]) for i in range(len(in_ch)-1)]).double()
    self.pro_fc = nn.Linear(96, 256)
    
    self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])  # layersize=4

    self.task = task

  def batch_graph(self, d):
    fa, fb, ga, gb, n = d
    scope, fal, fbl, gal, gbl = [], [], [], [], []
    na, nb = 0,0
    # print(f'fa shape: {fa.shape}, fb shpae:{fb.shape}, ga shape: {ga.shape}, gb shape:{gb.shape}, n shape:{n.shape}')
    for i in range(n.shape[0]):
      # print(n[i], n[i][0])
      an = int(n[i][0][0].item())  # atom num
      bn = int(n[i][0][1].item())  # bond num

      fal.append(fa[i,:an,:])
      fbl.append(fb[i,:bn,:])
      gal.append(ga[i,:an,:]+na)
      gbl.append(gb[i,:bn,:]+nb)
      scope.append((na,an))

      na += an
      nb += bn
    
    fas = Variable(torch.cat(fal, 0))
    fbs = Variable(torch.cat(fbl, 0))
    ags = Variable(torch.cat(gal, 0).long())
    bgs = Variable(torch.cat(gbl, 0).long())
    return fas, fbs, ags, bgs, scope
  
  def index_select(self, source, dim, index_):
    suffix_dim = source.size()[1:]
    final_size = index_.size() + suffix_dim
    target = source.index_select(dim, index_.view(-1))
    return target.view(final_size)


  def mpnn(self, d):
    fas, fbs, ags, bgs, scope = self.batch_graph(d)
    b = self.com_Wi(fbs)
    m = F.relu(b)            # message

    for i in range(self.com_net_depth-1):
      nm = self.index_select(m, 0, bgs)
      nm = nm.sum(dim=1)
      nm = self.com_Wh(nm)
      m = F.relu(b+nm)
    
    nm = self.index_select(m, 0, ags)
    nm = nm.sum(dim=1)
    a = torch.cat([fas, nm], dim=1)
    a = F.relu(self.com_Wo(a))
    x = torch.stack([torch.mean(a.narrow(0,start,g_size),0) for start, g_size in scope],0)
    return x


  def cnn(self, p):
    p = p.double()
    for l in self.pro_cnn:
      p = F.relu(l(p))
    p = F.adaptive_max_pool1d(p, output_size=1)
    p = p.view(p.size(0), -1)
    p = self.pro_fc(p.float())
    return p

  def forward(self, com, pro):
    com = self.mpnn(com)
    pro = self.cnn(pro)
    # print(com.shape, pro.shape)
    x = torch.cat((com, pro), 1)
    for i, l in enumerate(self.predictor):
      if i == (len(self.predictor)-1):
        if self.task == 'classification':
          x = torch.sigmoid(l(x))
        else:
          x = l(x)
      else:
        x = F.relu(self.dropout(l(x)))
    return x
