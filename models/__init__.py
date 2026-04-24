from typing import List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from .dcdti import DeepConvDTI
from .dpdta import DeepDTA
from .mdprd import MDeePred
from .dp import DeepPurpose
from .perceivercpi import PerceiverCPI
# from .smtdta import SMTDTA
from .model_MoE import DTI_Sparse_MoE

from .embedding import emb_dcdti, emb_dp, emb_dpdta, return_embedding

def _replace_batchnorm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            replacement = nn.GroupNorm(1, child.num_features, eps=child.eps, affine=True)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight.data)
                    replacement.bias.copy_(child.bias.data)
            setattr(module, name, replacement)
        else:
            _replace_batchnorm(child)
    return module


def build_model(name, task, com_len=100, pro_len=1000):
    if name == 'dcdti':
        return _replace_batchnorm(DeepConvDTI(1, task))
    elif name == 'dpdta':
        return _replace_batchnorm(DeepDTA(85, 1200, 1, task))  # 或用参数决定
    elif name == 'mdprd':
        return _replace_batchnorm(MDeePred(1024, 1, task))
    elif name == 'gifdti':
        from .gifdti import CNNFormerDTI
        drug_dict = {'max_len': com_len, 'encoder_dim': 256, 'embeding_dim': 256, 'embeding_num': 65, 'num_layers': 3, 'conv_kernel_size': 5, 'feed_forward_expansion_factor': 4, 'num_attention_heads': 8, 'attention_dropout_p': 0.1, 'conv_dropout_p': 0.1, 'predict_dropout_prob': 0.1}
        protein_dict = {'max_len': pro_len, 'encoder_dim': 256, 'embeding_dim': 256, 'embeding_num': 26, 'num_layers': 3, 'conv_kernel_size': 5, 'feed_forward_expansion_factor': 4, 'num_attention_heads': 8, 'attention_dropout_p': 0.1, 'conv_dropout_p': 0.1}
        return _replace_batchnorm(CNNFormerDTI(drug_dict, protein_dict))
    elif name == 'dp':
        return _replace_batchnorm(DeepPurpose([256+256, 1024, 1024, 512, 1], [26,32,64,96], [4,8,12], task))

    elif name == 'perceivercpi':
        # FIX 3: atom_dim=5, bond_dim=3 matches moe_dataset.py's inline RDKit graph
        # extraction: atoms=[AtomicNum,Degree,FormalCharge,NumExplicitHs,IsAromatic]
        # bonds=[BondTypeAsDouble, IsInRing, IsConjugated]  → cat dim = 5+3 = 8 ≠ 86
        return _replace_batchnorm(PerceiverCPI(atom_dim=5, bond_dim=3, prot_seq_len=pro_len, output_dim=1, task=task))
    elif name == 'smtdta':
        pass

# data-preprocessing and saving, run sub-classifiers, run ensemble

class GlobalMaxPooling1D(nn.Module):
  def __init__(self):
      super(GlobalMaxPooling1D, self).__init__()
      self.step_axis = 1

  def forward(self, input):
      return torch.max(input, axis=self.step_axis).values
