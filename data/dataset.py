import os
import re
import json
import pickle
import itertools

from bidict import bidict
from itertools import product
from collections import defaultdict


from random import Random
from typing import Dict, Iterator, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from .utils import *

import torch
from torch_geometric.data import Data, InMemoryDataset

PROTEIN_FEATURES = ['AAC', 'PAAC', 'DDE']

# code from DeepDTA:
CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, 
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, 
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, 
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, 
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, 
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, 
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, 
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62, "@": 63, "/": 64, "\\": 65}

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }




PRO_3GRAM_DICT = {}
AMINO = ''.join(CHARPROTSET.keys())
for idx, item in enumerate(product('-'+AMINO, AMINO, AMINO+'=')):
    PRO_3GRAM_DICT[''.join(item)] = idx


# input -> dataset (e.g.: kiba, davis ...)
# preprocess: process csv and save different data types and save them into /dataset/NAME/feature.pt
# when call the data, call 
class CPIDataset(InMemoryDataset):
    # def __init__(self, root, dataset, MAX_SMI_LEN, MAX_SEQ_LEN, label_type, mode):
    def __init__(self, root, dataset, label_type, mode, subset_size=None):
        super(InMemoryDataset, self).__init__()
        self.root = root
        self.dataset = dataset
        self.lig_dic = {}
        self.pro_dic = {}
        self.datapoint_dic = {}
        self.lig_mapping = {}
        self.pro_mapping = {}
        self.label_type = label_type # kd / catds / label / int

        self.com_feat_dir = f'{self.root}/dataset/{self.dataset}/com_feat'
        self.pro_feat_dir = f'{self.root}/dataset/{self.dataset}/pro_feat'

        # process the csv file
        datapath = os.path.join(self.root, 'dataset', self.dataset, 'data.csv')
        df = pd.read_csv(datapath)
        if subset_size is not None:
            # Consistent sampling for debug mode
            df = df.sample(n=min(subset_size, len(df)), random_state=42).reset_index(drop=True)
        self.ori_csv = df

        self.lig = self.ori_csv['lig']
        self.pro = self.ori_csv['pro']
        self.smi = self.ori_csv['smi']
        self.seq = self.ori_csv['seq']
        self.label = self.ori_csv[self.label_type]

        # whether generate/load dataset

        for i, r in self.ori_csv.iterrows():
            if r.lig not in self.lig_dic.keys():
                self.lig_mapping[len(self.lig_dic)] = r.lig
                self.lig_dic[r.lig] = r.smi
            if r.pro not in self.pro_dic.keys():
                self.pro_mapping[len(self.pro_dic)] = r.pro
                self.pro_dic[r.pro] = r.seq
            self.datapoint_dic[i] = (r.lig, r.pro)
        
        # if MAX_SMI_LEN != None:
        #     self.MAX_SMI_LEN = MAX_SMI_LEN
        # else:
        #     self.MAX_SMI_LEN = max([len(i) for i in list(self.lig_dic.values())])
        # if MAX_SEQ_LEN != None:
        #     self.MAX_SEQ_LEN = MAX_SEQ_LEN
        # else:
        #     self.MAX_SEQ_LEN = max([len(i) for i in list(self.pro_dic.values())])

        self.lig_mapping = bidict(self.lig_mapping)
        self.pro_mapping = bidict(self.pro_mapping)
        self.datapoint_dic = bidict(self.datapoint_dic)
        
        return

    @property
    def num_lig(self) -> int:
        return len(self.lig_dic)

    @property
    def num_pro(self) -> int:
        return len(self.pro_dic)

    @property
    def num_interaction(self) -> int:
        return len(self.label)

    def split_data(self, split_rule, N):

        ''' according to the split rule and N, save the splits into npy file, e.g. random_5.npy'''

        if split_rule == 'random':
            ids = np.arange(self.num_interaction)
            np.random.shuffle(ids)
            np.save(f'{self.root}/dataset/{self.dataset}/{split_rule}_{N}.npy', np.array_split(ids,N))

    def process_molecule(self, MAX_SMI_LEN, cf):

        ''' process molecules with the pre-defined feature types
            
            pre-defined molecule features:
            1. smiles string one-hot encoding
            2. molecular graph
            3. fingerprints
        '''

        # if self.mode == 'load':
        # if len(os.listdir(self.com_feat_dir)) == 4 and all([os.path.getsize(os.path.join(self.com_feat_dir,file)) for file in os.listdir(self.com_feat_dir)]):
        #     print("All compound features have been generated!")
        # else:

        os.makedirs(self.com_feat_dir, exist_ok=True)

        if cf in ['fp_1024', 'fp_2048']:
            self.generate_fp(cf)
        elif cf == 'graph':
            self.generate_mol_graph()
        elif cf == 'smi_enc':
            self.generate_smi_enc(MAX_SMI_LEN)
        elif cf == 'subgraph':                     # cpi-prediction
            self.generate_subgraph()
        elif cf == 'mpnn':
            self.generate_mpnn_feature(MAX_SMI_LEN)

    def generate_mpnn_feature(self, MAX_SMI_LEN=100):
        mpnn = []
        graphs = []
        max_atoms = 0
        max_bonds = 0

        # Pass 1: Parse all SMILES and find the global maximum dimensions
        for com, smi in self.lig_dic.items():
            padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
            fatoms, fbonds = [], [padding]
            in_bonds, all_bonds = [], [(-1, -1)]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise RuntimeError(f"[Dataset] Invalid SMILES for MPNN featurization: com_id={com} smiles={smi!r}")
            Chem.Kekulize(mol, clearAromaticFlags=True)
            n_atoms = mol.GetNumAtoms()
            if n_atoms <= 0:
                raise RuntimeError(f"[Dataset] Empty molecule after parsing: com_id={com} smiles={smi!r}")

            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx()
                y = a2.GetIdx()

                b = len(all_bonds)
                all_bonds.append((x, y))
                fbonds.append(torch.cat([fatoms[x], bond_features(bond)], 0))
                in_bonds[y].append(b)

                b = len(all_bonds)
                all_bonds.append((y, x))
                fbonds.append(torch.cat([fatoms[y], bond_features(bond)], 0))
                in_bonds[x].append(b)

            total_bonds = len(all_bonds)
            
            # Update global max dimensions
            max_atoms = max(max_atoms, n_atoms)
            max_bonds = max(max_bonds, total_bonds)
            
            fatoms = torch.stack(fatoms, 0)
            fbonds = torch.stack(fbonds, 0)
            agraph = torch.zeros(n_atoms, 6).long()  # 6 is the max number of bond
            bgraph = torch.zeros(total_bonds, 6).long()
            for a in range(n_atoms):
                for i, b in enumerate(in_bonds[a]):
                    agraph[a, i] = b

            for b1 in range(1, total_bonds):
                x, y = all_bonds[b1]
                for i, b2 in enumerate(in_bonds[x]):
                    if all_bonds[b2][0] != y:
                        bgraph[b1, i] = b2
                        
            graphs.append((fatoms, fbonds, agraph, bgraph, n_atoms, total_bonds))

        # Pass 2: Pad using the true maximum dimensions
        for fatoms, fbonds, agraph, bgraph, Natom, Nbond in graphs:
            atoms_completion_num = max_atoms - fatoms.shape[0]
            bonds_completion_num = max_bonds - fbonds.shape[0]

            fatoms_dim = fatoms.shape[1]
            fbonds_dim = fbonds.shape[1]
            fatoms = torch.cat([fatoms, torch.zeros(atoms_completion_num, fatoms_dim)], 0)
            fbonds = torch.cat([fbonds, torch.zeros(bonds_completion_num, fbonds_dim)], 0)
            agraph = torch.cat([agraph.float(), torch.zeros(atoms_completion_num, 6)], 0)
            bgraph = torch.cat([bgraph.float(), torch.zeros(bonds_completion_num, 6)], 0)
            shape_tensor = torch.Tensor([Natom, Nbond]).view(1,-1)
            mpnn.append([fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor.float()])
            
        mpnn = np.array([[t.numpy() for t in row] for row in mpnn], dtype=object)
        np.save(f'{self.com_feat_dir}/dp_mpnn.npy', mpnn)
        return


    def generate_fp(self, fp_type):
        fp_len = int(fp_type.split('_')[1])
        fps = []
        from rdkit.Chem import rdFingerprintGenerator
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_len)
        for com, smi in self.lig_dic.items():
            mol = Chem.MolFromSmiles(smi)
            fp = np.array(generator.GetFingerprint(mol), dtype=float)
            fps.append(fp)
        fps = np.stack(fps)
        np.save(f'{self.com_feat_dir}/fps_{fp_len}.npy', fps)
        return

    def generate_mol_graph(self):
        graphs = []
        for com, smi in self.lig_dic.items():
            mol = Chem.MolFromSmiles(smi)
            g = build_mol_graph(mol)
            graphs.append(g)
        torch.save(graphs, f'{self.com_feat_dir}/graphs.pt')
        return

    def generate_subgraph(self):
        atom_dict = defaultdict(lambda: len(atom_dict))
        bond_dict = defaultdict(lambda: len(bond_dict))
        edge_dict = defaultdict(lambda: len(edge_dict))
        fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
        coms, adjs = [], []

        for com, smi in self.lig_dic.items():
        # for i_, r_ in self.ori_csv.iterrows():
            # com = r_.lig
            # smi = r_.smi 
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            atoms = [(a.GetSymbol(), 'aromatic') if a.GetIsAromatic() else a.GetSymbol() for a in mol.GetAtoms()]
            atoms = np.array([atom_dict[a] for a in atoms])
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            coms.append(extract_fingerprints(atoms, i_jbond_dict, edge_dict, fingerprint_dict))
            adjs.append(np.array(Chem.GetAdjacencyMatrix(mol)))
            # if len(atoms) != len(np.array(Chem.GetAdjacencyMatrix(mol))):
            #     print(smi)
            #     print(o)
        np.save(f'{self.com_feat_dir}/cpi_com_graph_fps.npy', np.array(coms, dtype=object), allow_pickle=True)
        np.save(f'{self.com_feat_dir}/cpi_com_graph_adj.npy', np.array(adjs, dtype=object), allow_pickle=True)
        with open(f'{self.com_feat_dir}/cpi_fp_dict.pickle', 'wb') as f:
            pickle.dump(dict(fingerprint_dict), f)
        with open(f'{self.com_feat_dir}/cpi_atom_dict.pickle', 'wb') as f:
            pickle.dump(dict(atom_dict), f)
        with open(f'{self.com_feat_dir}/cpi_edge_dict.pickle', 'wb') as f:
            pickle.dump(dict(edge_dict), f)
        with open(f'{self.com_feat_dir}/cpi_bond_dict.pickle', 'wb') as f:
            pickle.dump(dict(bond_dict), f)
        return 

    def generate_smi_enc(self, MAX_SMI_LEN):         
        # can_smiles = {}
        smiles_encodings = []
        # smiles_one_hot_encodings = []

        for com, smi in self.lig_dic.items():
            mol = Chem.MolFromSmiles(smi)
            can_smi = Chem.MolToSmiles(mol)
            # one_hot_smi = build_one_hot_enc(can_smi, MAX_SMI_LEN, CHARCANSMISET)
            smi_enc = build_seq_enc(can_smi, CHARCANSMISET)
            # can_smiles[com] = can_smi
            # smiles_one_hot_encodings.append(one_hot_smi)
            smiles_encodings.append(smi_enc)
            
        smiles_encodings = np.array(smiles_encodings, dtype=object)

        # with open(f'{self.com_feat_dir}/can_smiles.json','w') as f:
            # f.write(json.dumps(can_smiles))
        np.save(f'{self.com_feat_dir}/smiles_encodings_{MAX_SMI_LEN}.npy', smiles_encodings)
        return

    def get_fp(self, fp_len):
        fp = np.load(f'{self.com_feat_dir}/fps_{fp_len}.npy', allow_pickle=True)
        return fp
    
    @property
    def get_can_smi(self):
        smiles_dic = json.load(open(f'{self.com_feat_dir}/can_smiles.json'))
        return smiles_dic

    # @property
    def get_smi_enc(self, MAX_SMI_LEN):
        smi_enc = np.load(f'{self.com_feat_dir}/smiles_encodings_{MAX_SMI_LEN}.npy', allow_pickle=True)
        # smi_enc = np.load(f'{self.com_feat_dir}/smiles_encodings.npy', allow_pickle=True)
        return smi_enc

    @property
    def get_mol_graph(self):
        graphs = torch.load(f'{self.com_feat_dir}/graphs.pt')
        return graphs
    
    @property
    def get_subgraph(self):

        ''' !!! ATTENTION !!!
            This  version supposes that there are no new compounds 
            (that all the compounds have been generated as graphs and saved in the files below)
        '''
        coms = np.load(f'{self.com_feat_dir}/cpi_com_graph_fps.npy', allow_pickle=True).tolist()
        adjs = np.load(f'{self.com_feat_dir}/cpi_com_graph_adj.npy', allow_pickle=True).tolist()
        return coms, adjs
    
    @property
    def get_mpnn_feature(self):
        mpnn = np.load(f'{self.com_feat_dir}/dp_mpnn.npy', allow_pickle=True)
        return mpnn

    @property
    def get_cpi_fp_dict(self):
        with open(f'{self.com_feat_dir}/cpi_fp_dict.pickle', 'rb') as f:
            return pickle.load(f)

    def process_protein(self, MAX_SEQ_LEN, pf):

        ''' process proteins with the pre-defined feature types
            
            pre-defined protein features:
            1. sequence string one-hot encoding
            2. ACC
            3. PACC
            4. DDE
            5. N-Gram
            6. mdprd 5 feat map
        '''

        # if len(os.listdir(self.pro_feat_dir)) == 7 and all([os.path.getsize(os.path.join(self.pro_feat_dir,file)) for file in os.listdir(self.pro_feat_dir)]):
            # print("All protein features have been generated!")
        # else:

        os.makedirs(self.pro_feat_dir, exist_ok=True)

        if pf == 'seq_enc':
            self.generate_seq_enc(MAX_SEQ_LEN)
        elif pf == 'mdprd':
            self.generate_mdprd_pro_feat(500)
        elif pf == 'word':                          # cpi-prediction
            amino = "IECLWAHSPMDVGRYFNQKTUX"
            self.generate_word(amino)
        elif pf == 'dp':
            self.generate_dp_pro_feat()
        
        # sequneces = {}
        # seq_one_hot_encodings = []
        # seq_encodings = []
        # fasta =""
        # three_grams = []

        # for pro, seq in self.pro_dic.items():
        #     sequneces[pro] = seq
        #     one_hot_seq = build_one_hot_enc(seq, self.MAX_SEQ_LEN, CHARPROTSET)
        #     seq_one_hot_encodings.append(one_hot_seq)
        #     seq_enc = build_seq_enc(seq, CHARPROTSET)
        #     seq_encodings.append(seq_enc)
        #     fasta += f'>xxx|{pro}|xxx\n{seq}\n'
        #     gram3 = build_ngram(seq, 3, PRO_3GRAM_DICT)
        #     three_grams.append(gram3)

        # fasta_file = f'{self.pro_feat_dir}/sequences.fasta'
        # with open(fasta_file,'w') as f:
        #     f.write(fasta)
        # with open(f'{self.pro_feat_dir}/sequences.json','w') as f:
        #     f.write(json.dumps(sequneces))
        
        # seq_encodings = np.array(seq_encodings, dtype=object)
        # np.save(f'{self.pro_feat_dir}/three_grams.npy', three_grams)
        # np.save(f'{self.pro_feat_dir}/sequences_encodings.npy', seq_encodings)
        
        # for pro_feat in PROTEIN_FEATURES:
        #     pro_feat_out_file = f'{self.pro_feat_dir}/{pro_feat}.csv'
        #     os.system(f'python {self.root}/support/iLearn/iLearn-protein-basic.py --file {fasta_file} --method {pro_feat} --format csv --out {pro_feat_out_file}')
        return
    
    def generate_seq_enc(self, MAX_SEQ_LEN):
        sequneces = {}
        seq_encodings = []
        for pro, seq in self.pro_dic.items():
            sequneces[pro] = seq
            seq_enc = build_seq_enc(seq, CHARPROTSET)
            seq_encodings.append(seq_enc)
        seq_encodings = np.array(seq_encodings, dtype=object)
        np.save(f'{self.pro_feat_dir}/sequences_encodings_{MAX_SEQ_LEN}.npy', seq_encodings)
        return

    def generate_dp_pro_feat(self):
        dp_pro = []
        for pro, seq in self.pro_dic.items():
            dp_pro.append([CHARPROTSET[i] for i in seq])
        dp_pro = np.array(dp_pro, dtype=object)
        np.save(f'{self.pro_feat_dir}/dp_pro.npy', dp_pro)
        return

    def generate_mdprd_pro_feat(self, size):

        pro_seq_dic = self.pro_dic
        fil_aa_list = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]    
        to_remove = re.compile(r'B*O*U*X*Z*')
        tar_feat_max_dict = {"sequencematrix": 210.0, 'ZHAC000103': 2.95,
                            'GRAR740104': 215.0, 'SIMK990101': 0.732, 'blosum62': 11.0}
        pro_feat = []
        for pro, seq in pro_seq_dic.items():
            seq = re.sub(to_remove, '', seq)
            pro_all_channel_features = []
            for feat in tar_feat_max_dict.keys():
                pro_all_channel_features.append(return_single_pro_feat_matrix(seq, feat, fil_aa_list, size) / tar_feat_max_dict[feat])
            pro_feat.append(torch.tensor(np.asarray(
                pro_all_channel_features, dtype=float).reshape(1, len(tar_feat_max_dict), 500, 500)).type(torch.FloatTensor))
        torch.save(torch.cat(pro_feat), f'{self.pro_feat_dir}/mdprd_pro.pth')
        return
    
    def generate_word(self, amino):
        protein_words_dict = {}
        for idx, item in enumerate(product('-'+amino, amino, '='+amino)):
            protein_words_dict[''.join(item)] = idx

        def split_sequence(sequence, ngram):
            sequence = '-' + sequence + '='
            words = [protein_words_dict[sequence[i:i+ngram]]
                    for i in range(len(sequence)-ngram+1)]
            return np.array(words)

        words = []        
        for pro, seq in self.pro_dic.items():
            words.append(split_sequence(seq, 3))
        np.save(f'{self.pro_feat_dir}/cpi_words.npy', np.array(words, dtype=object), allow_pickle=True)
        return idx+1

    @property
    def get_seq(self):
        seqs_dic = json.load(open(f'{self.pro_feat_dir}/sequences.json'))
        return seqs_dic

    @property
    def get_3gram(self):
        three_grams = np.load(f'{self.pro_feat_dir}/three_grams.npy', allow_pickle=True)
        return three_grams

    def get_seq_enc(self, MAX_SEQ_LEN):
        seq_enc = np.load(f'{self.pro_feat_dir}/sequences_encodings_{MAX_SEQ_LEN}.npy', allow_pickle=True)
        # seq_enc = np.load(f'{self.pro_feat_dir}/sequences_encodings.npy', allow_pickle=True)
        return seq_enc
    
    @property
    def get_word(self):
        words = np.load(f'{self.pro_feat_dir}/cpi_words.npy', allow_pickle=True).tolist()
        return words

    @property
    def get_dp(self):
        dp_pro = np.load(f'{self.pro_feat_dir}/dp_pro.npy', allow_pickle=True)
        return dp_pro

    @property
    def get_mdprd(self):
        pro_feat = torch.load(f'{self.pro_feat_dir}/mdprd_pro.pth')
        return pro_feat

    @property
    def get_AAC(self):
        AAC = pd.read_csv(f'{self.pro_feat_dir}/AAC.csv', header=None, index_col=0)
        AAC_dict = AAC.to_dict()
        return AAC_dict

    @property
    def get_PAAC(self):
        PAAC = pd.read_csv(f'{self.pro_feat_dir}/PAAC.csv', header=None, index_col=0)
        PAAC_dict = PAAC.to_dict()
        return PAAC_dict

    @property
    def get_DDE(self):
        DDE = pd.read_csv(f'{self.pro_feat_dir}/DDE.csv', header=None, index_col=0)
        DDE_dict = DDE.to_dict()
        return DDE_dict

    def get(self, idx):
        mol = self.lig[idx]
        pro = self.pro[idx]
        lab = self.label[idx]
        return mol, pro, lab

    def len(self):
        return len(self.label)


# customize CPI dataset, inherited from CPIDataset.
# when initialize, com_feat type and pro_type should be given,
# so that getitem can return the defined features of com/pro
class CustomCPIDataset(CPIDataset):
    def __init__(self, root, dataset, MAX_SMI_LEN, MAX_SEQ_LEN, label_type, cf, pf, mode):
        # super(CustomCPIDataset, self).__init__(root, dataset, MAX_SMI_LEN, MAX_SEQ_LEN, label_type, mode)
        super(CustomCPIDataset, self).__init__(root, dataset, label_type, mode)
        # print(f'MAX SMI len: {MAX_SMI_LEN}, MAX SEQ len: {MAX_SEQ_LEN}')

        # print(root, dataset, MAX_SMI_LEN, MAX_SEQ_LEN, label_type, cf, pf, mode)
        
        if mode == 'generate':
            self.process_molecule(MAX_SMI_LEN, cf)
            self.process_protein(MAX_SEQ_LEN, pf)

        self.cf = cf
        self.pf = pf

        if cf in ['fp_1024', 'fp_2048']:
            com_feat = self.get_fp(cf.split('_')[1])
        elif cf == 'smi_enc':
            com_feat = self.get_smi_enc(MAX_SMI_LEN)
        elif cf == 'subgraph':
            com_feat, com_adjs = self.get_subgraph            # [fps, adjs]
            # self.word_num = 10648
            self.fp_num = len(self.get_cpi_fp_dict) + 1
            self.amino = sorted(list("IECLWAHSPMDVGRYFNQKTUX"))
            self.word_num = len(self.amino)*(len(self.amino)+1)*(len(self.amino)+1)
        elif cf == 'mpnn':
            com_feat = self.get_mpnn_feature


        if pf == 'seq_enc':
            pro_feat = self.get_seq_enc(MAX_SEQ_LEN)
        elif pf == 'mdprd':
            pro_feat = self.get_mdprd
        elif pf == 'word':
            pro_feat = self.get_word                # np.array(words)
        elif pf == 'dp':
            pro_feat = self.get_dp

        com_feature = []
        pro_feature = []
        for i in range(self.num_interaction):
            cid = self.lig_mapping.inverse[self.lig[i]]
            pid = self.pro_mapping.inverse[self.pro[i]]

            c_feat_ = com_feat[cid]
            p_feat_ = pro_feat[pid]
            if cf == 'smi_enc':
                if len(c_feat_) < MAX_SMI_LEN:
                    com_feature.append(c_feat_+[0]*(MAX_SMI_LEN-len(c_feat_)))
                else:
                    com_feature.append(c_feat_[:MAX_SMI_LEN])
            elif cf == 'subgraph':
                # com_feature.append([com_feat[i], com_feat[i]])
                com_feature.append([c_feat_, com_adjs[cid]])
                # if len(com_feat[i]) != len(com_feat[i]):
                #     print(self.lig[i])
            else:
                com_feature.append(c_feat_)
            if pf in ['seq_enc', 'dp']:
                if len(p_feat_) < MAX_SEQ_LEN:
                    pro_feature.append(p_feat_+[0]*(MAX_SEQ_LEN-len(p_feat_)))
                else:
                    pro_feature.append(p_feat_[:MAX_SEQ_LEN])
            # elif pf == 'dp':
            #     if len(p_feat_) < MAX_SEQ_LEN:
            #         pro_feature.append(p_feat_+['?']*(MAX_SEQ_LEN-len(p_feat_)))
            #     else:
            #         pro_feature.append(p_feat_[:MAX_SEQ_LEN])
            else:
                pro_feature.append(p_feat_)
        if cf == 'subgraph':
            self.com_feat = com_feature
        elif type(com_feature[0]) == list:
            self.com_feat = np.array(com_feature)
        else:
            self.com_feat = com_feature

        if pf == 'word':
            self.pro_feat = [torch.LongTensor(p) for p in pro_feature]
        elif type(pro_feature) == list:
            self.pro_feat = np.array(pro_feature)    
        else:
            self.pro_feat = pro_feature 

        # self.data = np.concatenate((com_feature, pro_feature), axis=1)
        return

    def __getitem__(self, ind):
        if 'fp' in self.cf:
            com_feat = torch.tensor(self.com_feat[ind], dtype=torch.float32)
        elif self.cf == 'subgraph':
            com_feat = torch.LongTensor(self.com_feat[ind][0])
            com_adj = torch.FloatTensor(self.com_feat[ind][1])
            assert len(com_feat) == len(com_adj), f'{self.com_feat[ind]}'

        else:
            com_feat = self.com_feat[ind]
        lab = torch.tensor(self.label[ind], dtype=torch.float32)
        # pro_feat = torch.tensor(self.pro_feat[ind], dtype=torch.float32)
        if self.pf == 'dp':
            return {'com_id':self.lig[ind], 'pro_id':self.pro[ind], 'com_af':com_feat[0], 'com_bf':com_feat[1], 'com_ag':com_feat[2], 'com_bg':com_feat[3], 'com_abn':com_feat[4], 'pro_feat':np.eye(len(CHARPROTSET)+1)[self.pro_feat[ind]].T, 'label':lab}
        elif self.cf != 'subgraph':
            # return {'com_id':self.lig[ind], 'pro_id':self.pro[ind], 'com_feat':com_feat, 'pro_feat':self.pro_feat[ind].tolist(), 'label':lab}
            return {'com_id':self.lig[ind], 'pro_id':self.pro[ind], 'com_feat':com_feat, 'pro_feat':self.pro_feat[ind], 'label':lab}
        else:
            return {'com_id':self.lig[ind], 'pro_id':self.pro[ind], 'com_feat':com_feat, 'com_adj':com_adj, 'pro_feat':self.pro_feat[ind], 'label':lab}




# class DataLoader(torch.utils.data.DataLoader):

#     def __init__(
#         self,
#         dataset,
#         com_feat: str,
#         pro_feat: str,
#         batch_size: int = 1,
#         shuffle: bool = False,
#         follow_batch: Optional[List[str]] = None,
#         exclude_keys: Optional[List[str]] = None,
#         **kwargs,
#     ):
#         if 'collate_fn' in kwargs:
#             del kwargs['collate_fn']
        
#         '''  '''

#         super().__init()__


# def onek_encoding_unk(x, allowable_set):
#     if x not in allowable_set:
#         x = allowable_set[-1]
#     return list(map(lambda s: x == s, allowable_set))


# def atom_features(atom):
#     return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
#             + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
#             + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
#             + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
#             + [atom.GetIsAromatic()])

# def bond_features(bond):
#     bt = bond.GetBondType()
#     stereo = int(bond.GetStereo())
#     fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
#     fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
#     return torch.Tensor(fbond + fstereo)

# def build_one_hot_enc(line, max_len, chars):
# 	X = np.zeros((max_len, len(chars))) #+1
# 	for i, ch in enumerate(line[:max_len]):
# 		X[i, (chars[ch]-1)] = 1 
# 	return X #.tolist()

# def build_seq_enc(line, chars):
#     X = []
#     for i,ch in enumerate(line):
#         X.append(chars[ch])
#     return X

# def build_ngram(seq, N, dic):
#     seq = '-' + seq + '='
#     words = [dic[seq[i:i+N]]
#              for i in range(len(seq)-N+1)]
#     return np.array(words)

# def build_mol_graph(mol):
#     atom_features_list = []
#     for atom in mol.GetAtoms():
#         atom_feature = atom_to_feature_vector(atom)
#         atom_features_list.append(atom_feature)
#     x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

#     if len(mol.GetBonds()) <= 0:
#         num_bond_features = 3
#         edge_index = torch.empty((2,0), dtype=torch.long)
#         edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
#     else:
#         edges_list = []
#         edge_features_list = []
#         for bond in mol.GetBonds():
#             i = bond.GetBeginAtomIdx()
#             j = bond.GetEndAtomIdx()
#             edge_feature = bond_to_feature_vector(bond)

#             edges_list.append((i,j))
#             edge_features_list.append(edge_feature)
#             edges_list.append((j,i))
#             edge_features_list.append(edge_feature)

#         edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
#         edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

#         data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

#         return data

# def return_single_pro_feat_matrix(seq, feat, fil_aa_list, size):

#     # get pro feat dict
#     if feat == 'sequencematrix':        # [''.join(enc) for enc in itertools.product(fil_aa_list, repeat=2)]
#         aa_pair_encodings = {}
#         for i in fil_aa_list:
#             for p in range(len(fil_aa_list)):
#                 j = fil_aa_list[p]
#                 aa_pair_encodings[f'{i}{j}'] = aa_pair_encodings[f'{j}{i}'] = p+1
#     else:
#         aa_pair_encodings = json.load(open(f'support/encodings/{feat}.json', 'r'))
#     matrix = np.array([[aa_pair_encodings[f'{i}{j}'] for j in seq] for i in seq])   # j:index, i:column
    
#     # padding or chunking
#     len_m = len(matrix)
#     if len_m < size:
#         pad_l = (size - len_m) // 2
#         pad_r = pad_l + 1 if (size - len_m) % 2 else pad_l
#         matrix = np.pad(matrix,((pad_l, pad_r),(pad_l, pad_r)), mode='constant', constant_values=((0,0),(0,0)))
#     else:
#         matrix = matrix[:size, :size]
#     # print(len(matrix))
#     return matrix.flatten()

# def create_ijbonddict(mol, bond_dict):
#     i_jbond_dict = defaultdict(lambda: [])
#     for b in mol.GetBonds():
#         i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
#         bond = bond_dict[str(b.GetBondType())]
#         i_jbond_dict[i].append((j, bond))
#         i_jbond_dict[j].append((i, bond))
#     return i_jbond_dict

# def extract_fingerprints(atoms, i_jbond_dict, edge_dict, fingerprint_dict):
#     radius = 2
#     nodes = atoms
#     i_jedge_dict = i_jbond_dict

#     if len(atoms) == 1:
#         fingerprints = [fingerprint_dict[a] for a in atoms]
#     else:
#         for _ in range(radius):
#             fingerprints = []
#             for i, j_edge in i_jedge_dict.items():
#                 neighbors = [(nodes[j], edge) for j, edge in j_edge]
#                 fingerprint = (nodes[i], tuple(sorted(neighbors)))
#                 fingerprints.append(fingerprint_dict[fingerprint])
#                 for j, edge in j_edge:
#                     both_side = tuple(sorted((nodes[i], nodes[j])))
#                     edge = edge_dict[(both_side, edge)]
#     return np.array(fingerprints)

# def mapping_bool(value, l):
#     if value not in l:
#         return [False] * (len(l)-1) + [True]
#     else:
#         bool_l = [False] * len(l)
#         bool_l[l.index(value)] = True
#         return bool_l

