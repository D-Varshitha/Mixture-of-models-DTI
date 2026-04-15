import random
import os
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from datetime import datetime
from tqdm import tqdm
import timeit, pickle
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from torch import Tensor
import torch.nn.init as init
from typing import Tuple
from typing import Optional
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, recall_score,precision_recall_curve, auc



CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class View(nn.Module):
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            x = x.contiguous()

        return x.view(*self.shape)

class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)

class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)

class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 5,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_channels=in_channels,out_channels=in_channels*2,
            kernel_size=1,stride=1,padding=0,bias=True,),
            Swish(),
            nn.Conv1d(in_channels=in_channels*2,out_channels=in_channels*2,kernel_size=kernel_size,
                stride=1,padding=(kernel_size - 1) // 2,bias=True,
            ),
            GLU(dim=1),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True, ),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)

class RelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Tensor,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.to(torch.bool).unsqueeze(1).unsqueeze(2)
            score = score.masked_fill(mask, -1e9)
        attn = F.softmax(score, -1)
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, max_len: int = 1000):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model,max_len=max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)


class CNNformerBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            max_len: int = 1000
    ):
        super(CNNformerBlock, self).__init__()

        self.MHSA_model = MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                    max_len = max_len
                )
        self.CNN_model = ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    dropout_p=conv_dropout_p,
                )
        self.FF_model = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )

    def forward(self, inputs: Tensor,mask: Tensor) -> Tuple[Tensor, Tensor]:
        MHSA_out = self.MHSA_model(inputs,mask) + inputs
        CNN_out = self.CNN_model(MHSA_out) + MHSA_out
        FFout = 0.5 * self.FF_model(CNN_out) + 0.5 * CNN_out
        return FFout

class CNNformerEncoder(nn.Module):
    def __init__(
            self,
            max_len: int=1000,
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
    ):
        super(CNNformerEncoder, self).__init__()
        self.CNNformerlayers = nn.ModuleList([CNNformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            max_len = max_len
        ) for _ in range(num_layers)])

        self.FFlayers = nn.ModuleList([FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        ) for _ in range(num_layers)])

    def forward(self, inputs: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        CNNformeroutputs = inputs
        for num in range(len(self.CNNformerlayers)):
            FF_output = 0.5 * self.FFlayers[num](CNNformeroutputs) + 0.5 * CNNformeroutputs
            CNNformeroutputs = self.CNNformerlayers[num](FF_output,mask)
        return CNNformeroutputs

class CNNFormerDTI(nn.Module):
    def __init__(self, drug_dict, protein_dict, task='classification'):
        super(CNNFormerDTI, self).__init__()
        self.task = task
        self.protein_embed = nn.Embedding(protein_dict['embeding_num'], drug_dict['embeding_dim'], padding_idx=0)
        self.drug_embed = nn.Embedding(drug_dict['embeding_num'], protein_dict['embeding_dim'], padding_idx=0)
        self.protein_F = CNNformerEncoder(
            max_len = 1000,
            encoder_dim=256,
            num_layers=3,
            num_attention_heads=8,
            feed_forward_expansion_factor=4,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=5)
        self.drug_F = CNNformerEncoder(
            max_len = 100,
            encoder_dim=256,
            num_layers=3,
            num_attention_heads=8,
            feed_forward_expansion_factor=4,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=5)
        self.protein_attention_layer = nn.Linear(protein_dict["encoder_dim"], protein_dict["encoder_dim"])
        self.drug_attention_layer = nn.Linear(drug_dict["encoder_dim"], drug_dict["encoder_dim"] )
        self.protein_key_layer = nn.Linear(protein_dict["encoder_dim"], protein_dict["encoder_dim"])
        self.drug_key_layer = nn.Linear(drug_dict["encoder_dim"], drug_dict["encoder_dim"])
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(drug_dict["encoder_dim"] * 6, 1024)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout3 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def _masked_pool(self, features, pad_mask):
        valid = (~pad_mask).unsqueeze(-1).float()
        masked_max = features.masked_fill(pad_mask.unsqueeze(-1), float('-inf')).amax(dim=1)
        masked_max = torch.nan_to_num(masked_max, nan=0.0, posinf=0.0, neginf=0.0)
        masked_avg = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return torch.cat([masked_max, masked_avg], dim=1)

    def forward(self, drug, protein, drug_mask, protein_mask):
        drug_mask = drug_mask.to(torch.bool)
        protein_mask = protein_mask.to(torch.bool)
        drug_embed = self.drug_embed(drug)
        protein_embed = self.protein_embed(protein)
        """Feature extractor"""
        drug_feature = self.drug_F(drug_embed, drug_mask)
        protein_feature = self.protein_F(protein_embed, protein_mask)

        """Attention block"""
        drug_attention_key = F.leaky_relu(self.drug_key_layer(drug_feature),0.01)
        protein_attention_key = F.leaky_relu(self.protein_key_layer(protein_feature),0.01)
        Attention_matrix = torch.tanh(torch.einsum('baf,bfc->bac', drug_attention_key, protein_attention_key.permute(0, 2, 1)))
        drug_attention = torch.tanh(torch.sum(Attention_matrix, 2))
        protein_attention = torch.tanh(torch.sum(Attention_matrix, 1))

        # Attention
        drug_feature_a = F.leaky_relu(self.drug_attention_layer(drug_feature), 0.01)
        protein_feature_a = F.leaky_relu(self.protein_attention_layer(protein_feature), 0.01)
        drug_feature_a = drug_feature_a * drug_attention.unsqueeze(2)
        protein_feature_a = protein_feature_a * protein_attention.unsqueeze(2)

        """"Predictor"""
        drug_feature_a = self._masked_pool(drug_feature_a, drug_mask)
        drug_feature = self._masked_pool(drug_feature, drug_mask)
        protein_feature_a = self._masked_pool(protein_feature_a, protein_mask)
        protein_feature = self._masked_pool(protein_feature, protein_mask)

        iner_f = torch.mul(drug_feature_a, protein_feature_a)
        pair = torch.cat([drug_feature, iner_f, protein_feature], dim=1)
        pair = self.dropout1(pair)
        fully1 = F.leaky_relu(self.fc1(pair),0.01)
        fully1 = self.dropout2(fully1)
        fully2 = F.leaky_relu(self.fc2(fully1),0.01)
        fully2 = self.dropout3(fully2)
        fully3 = F.leaky_relu(self.fc3(fully2),0.01)
        predict = self.out(fully3)
        return predict


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN,dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def kmer_encode(line, kmer_AAS, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = kmer_AAS[line[i:i+3]]
    return X

class CustomDataSet(Dataset):
    def __init__(self, df):
        self.ori_index = df.index.tolist()
        self.df = df.reset_index()

    def __getitem__(self, idx):
        return self.df.loc[idx, 'lig'], self.df.loc[idx, 'pro'], self.df.loc[idx, 'smi'], self.df.loc[idx, 'seq'], self.df.loc[idx, 'lab']

    def __len__(self):
        return len(self.df)

def collate_fn(batch_data):
    N = len(batch_data)
    # drug_ids, protein_ids = [],[]
    compound_max = 100
    protein_max = 1000
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    compound_mask = torch.zeros((N, compound_max))
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    protein_mask = torch.zeros((N, protein_max))
    labels_new = torch.zeros(N, dtype=torch.long)
    coms = []
    pros = []
    for num, data in enumerate(batch_data):
        com, pro, smi, seq, label = data

        smiles_len = len(smi)
        compoundint = torch.from_numpy(label_smiles(smi, CHARISOSMISET, compound_max))
        compound_new[num] = compoundint
        if smiles_len > compound_max:
            compound_mask[num, :] = 1
        else:
            compound_mask[num, :smiles_len] = 1

        pro_len = len(seq)
        proteinint = torch.from_numpy(label_sequence(seq, CHARPROTSET, protein_max))
        # proteinint = torch.from_numpy(kmer_encode(seq, kmer_AAS, protein_max))
        protein_new[num] = proteinint
        if pro_len > protein_max:
            protein_mask[num, :] = 1
        else:
            protein_mask[num, :pro_len] = 1

        labels_new[num] = int(float(label))
        coms.append(com)
        pros.append(pro)

    return (coms, pros, compound_new, protein_new, compound_mask, protein_mask, labels_new)

class collate_class():
    def __init__(self, dict, compound_max = 100,protein_max=1000):
        self.protein_max = protein_max
        self.compound_max = compound_max
        self.word_dict = dict

    def split_sequence(self, sequence, ngram=3,max_lengh = 1000):
        words = [int(float(self.word_dict[sequence[i:i + ngram]]))
                 for i in range(len(sequence) - ngram + 1)]
        return np.array(words[:max_lengh])

    def __call__(self, batch_data):
        N = len(batch_data)
        compound_new = torch.zeros((N, self.compound_max), dtype=torch.long)
        compound_mask = torch.zeros((N, self.compound_max))
        protein_new = torch.zeros((N, self.protein_max), dtype=torch.long)
        protein_mask = torch.zeros((N, self.protein_max))
        labels_new = torch.zeros(N, dtype=torch.long)
        coms = []
        pros = []
        for num, data in enumerate(batch_data):
            com, pro, smi, seq, label = data

            smiles_len = len(smi)
            compoundint = torch.from_numpy(label_smiles(smi, CHARISOSMISET, self.compound_max))
            compound_new[num] = compoundint
            if smiles_len > self.compound_max:
                compound_mask[num, :] = 1
            else:
                compound_mask[num, :smiles_len] = 1

            pro_len = len(seq)
            proteinint = torch.from_numpy(self.split_sequence(seq))
            # proteinint = torch.from_numpy(kmer_encode(seq, kmer_AAS, protein_max))
            protein_new[num,:len(proteinint)] = proteinint
            if pro_len > self.protein_max:
                protein_mask[num, :] = 1
            else:
                protein_mask[num, :pro_len] = 1

            labels_new[num] = int(float(label))
            coms.append(com)
            pros.append(pro)
        return (coms, pros, compound_new, protein_new, compound_mask, protein_mask, labels_new)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def show_result(DATASET1, DATASET2, Loss_List, Accuracy_List,Precision_List,Recall_List,F1_score_List,AUC_List,AUPR_List):
    Loss_mean, Loss_std = np.mean(Loss_List), np.sqrt(np.var(Loss_List))
    Accuracy_mean, Accuracy_std = np.mean(Accuracy_List), np.sqrt(np.var(Accuracy_List))
    Precision_mean, Precision_var = np.mean(Precision_List), np.sqrt(np.var(Precision_List))
    Recall_mean, Recall_var = np.mean(Recall_List), np.sqrt(np.var(Recall_List))
    F1_score_mean, F1_score_var = np.mean(F1_score_List), np.sqrt(np.var(F1_score_List))
    AUC_mean, AUC_std = np.mean(AUC_List), np.sqrt(np.var(AUC_List))
    PRC_mean, PRC_std = np.mean(AUPR_List), np.sqrt(np.var(AUPR_List))
    print("The results on {} of {}:".format(DATASET1,DATASET2))
    with open(resultsavepath + 'results.txt', 'a') as f:
        f.write('{}:'.format(DATASET1) + '\n')
        f.write('Loss(std):{:.4f}({:.4f})'.format(Loss_mean, Loss_std) + '\n')
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('F1_score(std):{:.4f}({:.4f})'.format(F1_score_mean, F1_score_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_std) + '\n')
    print('Loss(std):{:.4f}({:.4f})'.format(Loss_mean, Loss_std))
    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('F1_score(std):{:.4f}({:.4f})'.format(F1_score_mean, F1_score_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_std))

def test_precess(model,pbar):
    model.eval()
    # test_losses = []
    test_df = pd.DataFrame()
    CID, PID, Y, P, S = [], [], [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            com, pro, compounds, adjs, proteins, masks, labels = data
            compounds = compounds.cuda()
            adjs = adjs.cuda()
            proteins = proteins.cuda()
            masks = masks.cuda()
            labels = labels.cuda()

            predicted_scores = model(compounds,adjs, proteins, masks)
            # loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            CID.extend(com)
            PID.extend(pro)
            # test_losses.append(loss.item())
    test_df['com'] = CID
    test_df['pro'] = PID
    test_df['pred'] = S
    test_df['lab'] = Y
    Precision = precision_score(Y, P)
    Reacll = recall_score(Y, P)
    F1_score = f1_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    # test_loss = np.average(test_losses) 
    return Y, P, Accuracy, Precision, Reacll, F1_score, AUC, PRC, test_df

def test_model(dataloader):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataloader)),
        total=len(dataloader))
    T, P, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test, test_df = \
        test_precess(model,test_pbar)\
    # count num of existing files in output folder
    # num_file = len([name for name in os.listdir('derived') if os.path.isfile(name)])
    test_df.to_csv(f'derived/result_gifdti.csv', index=False)
    return Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Davis', help='Dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    return args

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

if __name__ == "__main__":

    args = argument_parser()

    """select seed"""
    DATASET = args.dataset

    """init hyperparameters"""
    drug_dict = {'max_len': 100,
                 'encoder_dim': 256,
                 'embeding_dim': 256,
                 'embeding_num': 65,
                 'num_layers': 3,
                 'conv_kernel_size': 5,
                 'feed_forward_expansion_factor': 4,
                 'num_attention_heads': 8,
                 'attention_dropout_p': 0.1,
                 'conv_dropout_p': 0.1,
                 'predict_dropout_prob': 0.1
                 }

    protein_dict = {'max_len': 1000,
                    'encoder_dim': 256,
                    'embeding_dim': 256,
                    'embeding_num': 26,
                    'num_layers': 3,
                    'conv_kernel_size': 5,
                    'feed_forward_expansion_factor': 4,
                    'num_attention_heads': 8,
                    'attention_dropout_p': 0.1,
                    'conv_dropout_p': 0.1}
    
    """Load preprocessed data."""
    data_df = pd.read_csv(f'input/{DATASET}/data.csv')

    resultsavepath = "./ConFormerDTI/{}/".format(DATASET)

    """load protein embed matrix"""
    test_df = CustomDataSet(data_df)
    test_dataloader = DataLoader(test_df, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                    collate_fn=collate_fn)
    model = CNNFormerDTI(drug_dict, protein_dict).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("models/gifdti.pth"))

    Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test = \
        test_model(test_dataloader)
