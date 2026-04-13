from typing import List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from .dcdti import DeepConvDTI
from .dpdta import DeepDTA
from .mdprd import MDeePred
from .cpi import CPIPrediction
from .dp import DeepPurpose
# from .perceivercpi import PerceiverCPI
# from .smtdta import SMTDTA

from .embedding import emb_dcdti, emb_dp, emb_dpdta, return_embedding

def build_model(name, task, data=None):
    if name == 'dcdti':
        return DeepConvDTI(1, task)
    elif name == 'dpdta':
        return DeepDTA(85, 1200, 1, task)  # 或用参数决定
    elif name == 'mdprd':
        return MDeePred(1024, 1, task)
    elif name == 'cpi':
        return CPIPrediction(data.fp_num, data.word_num, 10, task)
    elif name == 'dp':
        return DeepPurpose([256+256, 1024, 1024, 512, 1], [26,32,64,96], [4,8,12], task)
    elif name == 'perceivercpi':
        pass
    elif name == 'smtdta':
        pass

# data-preprocessing and saving, run sub-classifiers, run ensemble

class GlobalMaxPooling1D(nn.Module):
  def __init__(self):
      super(GlobalMaxPooling1D, self).__init__()
      self.step_axis = 1

  def forward(self, input):
      return torch.max(input, axis=self.step_axis).values