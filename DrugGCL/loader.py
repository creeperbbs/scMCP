import math
import random
from copy import deepcopy
from torch.utils.data.distributed import DistributedSampler
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import rdkit
from rdkit import Chem
from torch_geometric.data import Data, Dataset, DataLoader
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
import torch.nn.functional as F
# allowable node and edge features
FEATURES_LIST = {
    'ATOM_LIST' : list(range(1, 119)),
    'FORMAL_CHARGE_LIST' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'CHIRALITY_LIST' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'HYBRIDIZATION_LIST' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'H_NUM_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'IMPLICIT_VALENCE_LIST' : [0, 1, 2, 3, 4, 5, 6],
    'DEGREE_LIST' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'BOND_LIST' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'BONDDIR_LIST' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}
ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC

]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def remove_subgraph(Graph, center, percent=0.2):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes) * percent))
    removed = []
    temp = [center]

    while len(removed) < num:
        neighbors = []
        if len(temp) < 1:
            break

        for n in temp:
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break

        temp = list(set(neighbors))
    return G, removed


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = pd.read_csv(csv_file)
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data

def bool_to_int(value):
    return 1 if value else 0


class MoleculeProcessor():
    def __init__(self, num_edge_type=5,mask_rate=0.25,mask_edge=0.25):
        
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.num_bond_direction = 3
    def process(self, batch):
        # print(batch)
        data_i_list, data_j_list = [], []
        for smiles in batch:  
            di, dj = self.process_single(smiles['text'])
            data_i_list.append(di)
            data_j_list.append(dj)
        
        return (
            Batch.from_data_list(data_i_list),
            Batch.from_data_list(data_j_list)
        )
    def process_single(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        # mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        #########################
        # Get the molecule info #
        #########################
        type_idx = []
        degree_idx = []
        hs_idx = [] 
        formal_charge_idx = [] 
        implicit_valence_idx = [] 
        chirality_idx = []
        hybridization_idx = []
        atomic_number = []
        for atom in atoms:
            type_idx.append(FEATURES_LIST['ATOM_LIST'].index(atom.GetAtomicNum()))
            degree_idx.append(FEATURES_LIST['DEGREE_LIST'].index(atom.GetDegree()))
            hs_idx.append(FEATURES_LIST['H_NUM_list'].index(atom.GetTotalNumHs()))
            formal_charge_idx.append(FEATURES_LIST['FORMAL_CHARGE_LIST'].index(atom.GetFormalCharge()))
            implicit_valence_idx.append(FEATURES_LIST['IMPLICIT_VALENCE_LIST'].index(atom.GetImplicitValence()))
            chirality_idx.append(FEATURES_LIST['CHIRALITY_LIST'].index(atom.GetChiralTag()))
            hybridization_idx.append(FEATURES_LIST['HYBRIDIZATION_LIST'].index(atom.GetHybridization()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x3 = torch.tensor(hybridization_idx, dtype=torch.long).view(-1, 1)
        x4 = torch.tensor(degree_idx, dtype=torch.long).view(-1, 1)
        x5 = torch.tensor(hs_idx, dtype=torch.long).view(-1, 1)
        x6 = torch.tensor(formal_charge_idx, dtype=torch.long).view(-1, 1)
        x7 = torch.tensor(implicit_valence_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2, x3,x4,x5,x6,x7], dim=-1)


        ####################
        # Subgraph Masking #
        ####################

        # Construct the original molecular graph from edges (bonds)
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        molGraph = nx.Graph(edges)

        # Get the graph for i and j after removing subgraphs
        start_i, start_j = random.sample(list(range(N)), 2)
        percent_i, percent_j = random.uniform(0, 0.2), random.uniform(0, 0.2)
        G_i, removed_i = remove_subgraph(molGraph, start_i, percent=percent_i)
        G_j, removed_j = remove_subgraph(molGraph, start_j, percent=percent_j)

        # Only consider bond still exist after removing subgraph
        row_i, col_i, row_j, col_j = [], [], [], []
        edge_feat_i, edge_feat_j = [], []
        G_i_edges = list(G_i.edges)
        G_j_edges = list(G_j.edges)

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir()),
                bool_to_int(bond.GetIsConjugated()),  
                bool_to_int(bond.IsInRing())
            ]
            if (start, end) in G_i_edges or (end, start) in G_i_edges:
                row_i += [start, end]
                col_i += [end, start]
                edge_feat_i.append(feature)
                edge_feat_i.append(feature)
            if (start, end) in G_j_edges or (end, start) in G_j_edges:
                row_j += [start, end]
                col_j += [end, start]
                edge_feat_j.append(feature)
                edge_feat_j.append(feature)

        edge_index_i = torch.tensor([row_i, col_i], dtype=torch.long)
        edge_attr_i = torch.tensor(np.array(edge_feat_i), dtype=torch.long)
        edge_index_j = torch.tensor([row_j, col_j], dtype=torch.long)
        edge_attr_j = torch.tensor(np.array(edge_feat_j), dtype=torch.long)
        num_nodes = x.size(0)

        ############################
        # Random Atom/Edge Masking #
        ############################

        atom_remain_indices_i = [i for i in range(N) if i not in removed_i]
        atom_remain_indices_j = [i for i in range(N) if i not in removed_j]

        num_mask_nodes_i = max([1, math.floor(0.25 * N) - len(removed_i)])
        num_mask_nodes_j = max([1, math.floor(0.25 * N) - len(removed_j)])
        mask_nodes_i = random.sample(atom_remain_indices_i, num_mask_nodes_i)
        mask_nodes_j = random.sample(atom_remain_indices_j, num_mask_nodes_j)

        x_i = deepcopy(x)
        mask_node_labels_list_i = []
        for atom_idx in range(N):
            if (atom_idx in mask_nodes_i) or (atom_idx in removed_i):
                mask_node_labels_list_i.append(x_i[atom_idx].detach().clone().view(1, -1))
                x_i[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0, len(FEATURES_LIST['HYBRIDIZATION_LIST'])-1,11,len(FEATURES_LIST['H_NUM_list']),6,len(FEATURES_LIST['IMPLICIT_VALENCE_LIST'])])
        # print(mask_node_labels_list_i)


        mask_node_label_i = torch.cat(mask_node_labels_list_i, dim=0)
        masked_atom_indices_i = torch.tensor(list(set(mask_nodes_i) | set(removed_i)))

        atom_type = F.one_hot(mask_node_label_i[:, 0], num_classes=len(ATOM_LIST)+1).float()
        atom_chirality = F.one_hot(mask_node_label_i[:, 1], num_classes=len(FEATURES_LIST['CHIRALITY_LIST'])).float()
        atom_hybridization = F.one_hot(mask_node_label_i[:, 2], num_classes=len(FEATURES_LIST['HYBRIDIZATION_LIST'])).float()
        atom_degree = F.one_hot(mask_node_label_i[:, 3], num_classes=len(FEATURES_LIST['DEGREE_LIST'])+1).float()
        atom_hs = F.one_hot(mask_node_label_i[:, 4], num_classes=len(FEATURES_LIST['H_NUM_list'])+1).float()
        atom_formal_charge = F.one_hot(mask_node_label_i[:, 5], num_classes=len(FEATURES_LIST['FORMAL_CHARGE_LIST'])+1).float()
        atom_implicit_valence = F.one_hot(mask_node_label_i[:, 6], num_classes=len(FEATURES_LIST['IMPLICIT_VALENCE_LIST'])+1).float()
        node_attr_label_i = torch.cat((atom_type,atom_chirality, atom_hybridization,atom_degree,atom_hs,atom_formal_charge,atom_implicit_valence), dim=1)  # 获取总特征 119+4+7个

        connected_edge_indices = []
        for bond_idx, (u, v) in enumerate(edge_index_i.cpu().numpy().T):
            for atom_idx in mask_nodes_i:
                if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                    connected_edge_indices.append(bond_idx)  

        if len(connected_edge_indices) > 0:
            mask_edge_labels_list = []
            for bond_idx in connected_edge_indices[::2]:
                mask_edge_labels_list.append(
                    edge_attr_i[bond_idx].view(1, -1))

            mask_edge_label_i = torch.cat(mask_edge_labels_list, dim=0)
            for bond_idx in connected_edge_indices:
                edge_attr_i[bond_idx] = torch.tensor(
                    [self.num_edge_type-1, 0, 2, 2])  

            connected_edge_indices_i = torch.tensor(
                connected_edge_indices[::2])
        else:
            mask_edge_label_i = torch.empty((0, 4)).to(torch.int64)
            connected_edge_indices_i = torch.tensor(
                connected_edge_indices).to(torch.int64)
        edge_type = F.one_hot(mask_edge_label_i[:, 0], num_classes=self.num_edge_type).float()
        bond_direction = F.one_hot(mask_edge_label_i[:, 1], num_classes=self.num_bond_direction).float()
        conjugated = F.one_hot(mask_edge_label_i[:, 2], num_classes=3).float()
        inring = F.one_hot(mask_edge_label_i[:, 3], num_classes=3).float()
        edge_attr_label_i = torch.cat((edge_type, bond_direction,conjugated,inring), dim=1)  #

        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i,node_attr_label=node_attr_label_i,masked_atom_indices=masked_atom_indices_i,connected_edge_indices=connected_edge_indices_i,edge_attr_label=edge_attr_label_i,mask_edge_label=mask_edge_label_i.float())

        x_j = deepcopy(x)
        mask_node_labels_list_j = []
        for atom_idx in range(N):
            if (atom_idx in mask_nodes_j) or (atom_idx in removed_j):
                mask_node_labels_list_j.append(x_j[atom_idx].detach().clone().view(1, -1))
                x_j[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0, len(FEATURES_LIST['HYBRIDIZATION_LIST'])-1,11,len(FEATURES_LIST['H_NUM_list']),6,len(FEATURES_LIST['IMPLICIT_VALENCE_LIST'])])

        mask_node_label_j = torch.cat(mask_node_labels_list_j, dim=0)
        masked_atom_indices_j = torch.tensor(list(set(mask_nodes_j) | set(removed_j)))

        atom_type = F.one_hot(mask_node_label_j[:, 0], num_classes=len(ATOM_LIST)+1).float()
        atom_chirality = F.one_hot(mask_node_label_j[:, 1], num_classes=len(FEATURES_LIST['CHIRALITY_LIST'])).float()
        atom_hybridization = F.one_hot(mask_node_label_j[:, 2], num_classes=len(FEATURES_LIST['HYBRIDIZATION_LIST'])).float()
        atom_degree = F.one_hot(mask_node_label_j[:, 3], num_classes=len(FEATURES_LIST['DEGREE_LIST'])+1).float()
        atom_hs = F.one_hot(mask_node_label_j[:, 4], num_classes=len(FEATURES_LIST['H_NUM_list'])+1).float()
        atom_formal_charge = F.one_hot(mask_node_label_j[:, 5], num_classes=len(FEATURES_LIST['FORMAL_CHARGE_LIST'])+1).float()
        atom_implicit_valence = F.one_hot(mask_node_label_j[:, 6], num_classes=len(FEATURES_LIST['IMPLICIT_VALENCE_LIST'])+1).float()
        node_attr_label_j = torch.cat((atom_type,atom_chirality, atom_hybridization,atom_degree,atom_hs,atom_formal_charge,atom_implicit_valence), dim=1)  

        connected_edge_indices = []
        for bond_idx, (u, v) in enumerate(edge_index_j.cpu().numpy().T):
            for atom_idx in mask_nodes_j:
                if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                    connected_edge_indices.append(bond_idx) 

        if len(connected_edge_indices) > 0:
            mask_edge_labels_list = []
            for bond_idx in connected_edge_indices[::2]:
                mask_edge_labels_list.append(
                    edge_attr_j[bond_idx].view(1, -1))

            mask_edge_label_j = torch.cat(mask_edge_labels_list, dim=0)
            for bond_idx in connected_edge_indices:
                edge_attr_j[bond_idx] = torch.tensor(
                    [self.num_edge_type-1, 0,2,2]) 

            connected_edge_indices_j = torch.tensor(
                connected_edge_indices[::2])
        else:
            mask_edge_label_j = torch.empty((0, 4)).to(torch.int64)
            connected_edge_indices_j = torch.tensor(
                connected_edge_indices).to(torch.int64)
        edge_type = F.one_hot(mask_edge_label_j[:, 0], num_classes=self.num_edge_type).float()
        bond_direction = F.one_hot(mask_edge_label_j[:, 1], num_classes=self.num_bond_direction).float()
        conjugated = F.one_hot(mask_edge_label_j[:, 2], num_classes=3).float()
        inring = F.one_hot(mask_edge_label_j[:, 3], num_classes=3).float()
        edge_attr_label_j = torch.cat((edge_type, bond_direction,conjugated,inring), dim=1)  # totally obtained 5+3+3+3 

        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j, node_attr_label=node_attr_label_j,
                      masked_atom_indices=masked_atom_indices_j, connected_edge_indices=connected_edge_indices_j,
                      edge_attr_label=edge_attr_label_j,mask_edge_label=mask_edge_label_j.float())

        return data_i, data_j

