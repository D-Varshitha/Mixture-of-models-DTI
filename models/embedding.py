import numpy as np
import torch
import torch.nn.functional as F


def emb_dp(self, com, pro):
  com = self.mpnn(com)
  pro = self.cnn(pro)
  # print(com.shape, pro.shape)
  x = torch.cat((com, pro), 1)
  for i, l in enumerate(self.predictor):
    if i == (len(self.predictor)-1):
      emb = x
      x = l(x)
    else:
      x = F.relu(self.dropout(l(x)))  
  return emb

def emb_dpdta(self, com, pro):
  com = self.com_emb(com)
  com = com.permute(0,2,1)
  pro = self.pro_emb(pro)
  pro = pro.permute(0,2,1)
  pro = self.pro_layer(pro).permute(0,2,1)
  com = self.com_layer(com).permute(0,2,1)
  pro = self.pro_max_pool(pro)
  com = self.com_max_pool(com)
  x = torch.cat((com, pro), 1)
  emb = self.int_layer(x)
  return emb

def emb_dcdti(self, com, pro):

  pro = self.pro_layer_init(pro)
  pro = pro.permute(0,2,1)

  pro_1 = self.pro_layer_cnn[0](pro).permute(0,2,1)
  pro_2 = self.pro_layer_cnn[1](pro).permute(0,2,1)
  pro_3 = self.pro_layer_cnn[2](pro).permute(0,2,1)
  pro_4 = self.pro_layer_cnn[3](pro).permute(0,2,1)
  pro_5 = self.pro_layer_cnn[4](pro).permute(0,2,1)

  pro_1 = self.max_pool(pro_1)
  pro_2 = self.max_pool(pro_2)
  pro_3 = self.max_pool(pro_3)
  pro_4 = self.max_pool(pro_4)
  pro_5 = self.max_pool(pro_5)

  pro = torch.cat((pro_1, pro_2, pro_3, pro_4, pro_5), 1)

  pro = self.pro_layer(pro)
  com = self.com_layer(com)

  x = torch.cat((com, pro), 1)
  x = self.int_layer[0](x)
  emb = self.int_layer[1](x)
  return emb

def return_embedding(model, generator, ty):
  embs = []
  labels = []
  
  if ty == 'dp':
    with torch.no_grad():
      for i, data in enumerate(generator, 0):
        cid, pid, com_af, com_bf, com_ag, com_bg, com_abn, pro, lab = data['com_id'], data['pro_id'], data['com_af'], data['com_bf'], data['com_ag'], data['com_bg'], data['com_abn'], data['pro_feat'], data['label']
        embs.append(model([com_af, com_bf, com_ag, com_bg, com_abn],pro).cpu().detach().numpy())
        labels.append(lab)
  elif ty != 'cpi':
    with torch.no_grad():
      for i, data in enumerate(generator, 0):
        cid, pid, com, pro, lab = data['com_id'], data['pro_id'], data['com_feat'], data['pro_feat'], data['label']
        # print(cid, pid, com, pro, lab)
        embs.append(model(com, pro).cpu().detach().numpy())
        labels.append(lab)
  else:
    with torch.no_grad():
      for i, data in enumerate(generator, 0):
        cid, pid, com, adj, pro, lab = data['com_id'], data['pro_id'], data['com_feat'], data['com_adj'], data['pro_feat'], data['label']
        embs.append(model(com, adj, pro).cpu().detach().numpy())
        labels.append(lab)
  return np.concatenate(embs), np.concatenate(labels)


  
  