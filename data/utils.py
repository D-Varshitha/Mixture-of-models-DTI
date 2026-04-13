import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from itertools import product
from collections import defaultdict
import json


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

def build_one_hot_enc(line, max_len, chars):
	X = np.zeros((max_len, len(chars))) #+1
	for i, ch in enumerate(line[:max_len]):
		X[i, (chars[ch]-1)] = 1 
	return X #.tolist()

def build_seq_enc(line, chars):
    X = []
    for i,ch in enumerate(line):
        X.append(chars[ch])
    return X

def build_ngram(seq, N, dic):
    seq = '-' + seq + '='
    words = [dic[seq[i:i+N]]
             for i in range(len(seq)-N+1)]
    return np.array(words)

def build_mol_graph(mol):
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    if len(mol.GetBonds()) <= 0:
        num_bond_features = 3
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i,j))
            edge_features_list.append(edge_feature)
            edges_list.append((j,i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

def return_single_pro_feat_matrix(seq, feat, fil_aa_list, size):

    # get pro feat dict
    if feat == 'sequencematrix':        # [''.join(enc) for enc in itertools.product(fil_aa_list, repeat=2)]
        aa_pair_encodings = {}
        for i in fil_aa_list:
            for p in range(len(fil_aa_list)):
                j = fil_aa_list[p]
                aa_pair_encodings[f'{i}{j}'] = aa_pair_encodings[f'{j}{i}'] = p+1
    else:
        aa_pair_encodings = json.load(open(f'support/encodings/{feat}.json', 'r'))
    matrix = np.array([[aa_pair_encodings[f'{i}{j}'] for j in seq] for i in seq])   # j:index, i:column
    
    # padding or chunking
    len_m = len(matrix)
    if len_m < size:
        pad_l = (size - len_m) // 2
        pad_r = pad_l + 1 if (size - len_m) % 2 else pad_l
        matrix = np.pad(matrix,((pad_l, pad_r),(pad_l, pad_r)), mode='constant', constant_values=((0,0),(0,0)))
    else:
        matrix = matrix[:size, :size]
    # print(len(matrix))
    return matrix.flatten()

def create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, edge_dict, fingerprint_dict):
    radius = 2
    nodes = atoms
    i_jedge_dict = i_jbond_dict

    if len(atoms) == 1:
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
    return np.array(fingerprints)

def mapping_bool(value, l):
    if value not in l:
        return [False] * (len(l)-1) + [True]
    else:
        bool_l = [False] * len(l)
        bool_l[l.index(value)] = True
        return bool_l
