
import torch
import torch.nn as nn
import torch.nn.functional as F

class MDeePred(nn.Module):    # com: fp_1024, pro: mdprd
  def __init__(self, fc_2, out_dim, task):
    super().__init__()
    self.com_layer = nn.Sequential(
      nn.Linear(1024,1024),
      nn.BatchNorm1d(num_features=1024),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(1024,1024),      
      nn.BatchNorm1d(num_features=1024),
      nn.ReLU(),
      nn.Dropout(0.1)
    )
    self.pro_layer_1 = nn.Sequential(          # CNNModuleInception
      nn.Conv2d(in_channels=5, out_channels=16, kernel_size=7, stride=3, padding=4),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    self.multi_pro_cnn = nn.ModuleList(        # VariableLengthFeatureDetector
       [nn.Sequential(            # branch 1x1
          nn.Conv2d(32, 64, bias=False, kernel_size=1),
          nn.BatchNorm2d(64, eps=0.001),
          nn.ReLU(inplace=True)
        ),
        nn.Sequential(            # branch 7x7
          nn.Conv2d(32, 32, bias=False, kernel_size=1),
          nn.BatchNorm2d(32, eps=0.001),
          nn.ReLU(inplace=True),
          nn.Conv2d(32, 64, bias=False, kernel_size=7, padding=3),
          nn.BatchNorm2d(64, eps=0.001),
          nn.ReLU(inplace=True)
        ),
        nn.Sequential(            # branch 5x5
          nn.Conv2d(32, 32, bias=False, kernel_size=1),
          nn.BatchNorm2d(32, eps=0.001),
          nn.ReLU(inplace=True),
          nn.Conv2d(32, 64, bias=False, kernel_size=5, padding=2),
          nn.BatchNorm2d(64, eps=0.001),
          nn.ReLU(inplace=True)
        ),
        nn.Sequential(            # branch 3x3
          nn.Conv2d(32, 64, bias=False, kernel_size=1),
          nn.BatchNorm2d(64, eps=0.001),
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 96, bias=False, kernel_size=3, padding=1),
          nn.BatchNorm2d(96, eps=0.001),
          nn.ReLU(inplace=True)
        ),        
        nn.Sequential(            # branch pool
          nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
          nn.Conv2d(32, 32, bias=False, kernel_size=1),
          nn.BatchNorm2d(32, eps=0.001),
          nn.ReLU(inplace=True)
      )]
    )
    self.pro_layer_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.pro_layer_3 = nn.Sequential(
      nn.Linear(320 * 21 * 21, 256),
      nn.ReLU())
    self.int_layer = nn.Sequential(
      nn.Linear(1024+256, 1024),
      nn.BatchNorm1d(num_features=1024),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(1024,fc_2),     # fc_2 = 1024 (davis) / 512 (kinome)
      nn.BatchNorm1d(num_features=fc_2),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(fc_2, out_dim)

    )
    self.task = task

  def forward(self, com, pro):
    com = self.com_layer(com)
    pro = self.pro_layer_1(pro)
    pro = torch.cat([self.multi_pro_cnn[i](pro) for i in range(5)],1)
    pro = self.pro_layer_2(pro).view(-1, 320 * 21 * 21)
    pro = self.pro_layer_3(pro)

    # print(com.shape, pro.shape)
    x = torch.cat((com, pro), 1)
    x = self.int_layer(x)
    # FIX 1: Output raw logits; BCEWithLogitsLoss handles sigmoid externally.
    return x
