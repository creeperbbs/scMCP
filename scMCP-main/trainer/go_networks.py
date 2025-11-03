import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor
import torch.nn as nn
from pathlib import Path
import networkx as nx
from tqdm import tqdm
import pickle
import sys, os
import requests
from torch_geometric.data import Data
from scipy import sparse



class GeneSimNetwork():
    '''f
    Represents a gene similarity network. Has the following attributes:
        - self.edge_list
        - self.gene_list
        - self.G
        - self.edge_index (Tensor, can be used to set G_go in model_initialize)
        - self.edge_weight (Tensor, can be used to set G_go_weight in model_initialize)
    '''
    def __init__(self, edge_list, gene_list, node_map):
        # Keeps edges if both source and target exist in node_map
        keep_edges = np.zeros(len(edge_list)).astype(bool)
        for i, (source, target, importance) in edge_list.iterrows():
            keep_edges[i] = source in node_map and target in node_map
        self.edge_list = edge_list[keep_edges]

        # Constructs NetworkX graph
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())
        self.gene_list = gene_list

        # Add any isolated nodes (represented in the tensor with self edges)
        for n in self.gene_list:
            added_nodes = []
            if n not in self.G.nodes():
                self.G.add_node(n)
                added_nodes.append(n)
        if len(added_nodes) > 0:
            print('The following nodes were not present in the edge list and have been added to the graph as self-edges:', added_nodes)
        
        # Convert graph edges to tensor
        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        
        # Convert edge weights to tensor
        edge_attr = nx.get_edge_attributes(self.G, 'importance') 
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)
import scipy.sparse as sp
def get_similarity_network(network_type, adata, threshold, k, gene_list, data_path, data_name, split, seed, train_gene_set_size):
    
    if network_type == 'co-express':
        df_out = get_coexpression_network_from_train(adata, threshold, int(k), data_path, data_name, split, seed, train_gene_set_size)
        
    elif network_type == 'go':
        df_jaccard = get_go_auto(gene_list, data_path, data_name)
        df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(int(k[0]) + 1,['importance'])).reset_index(drop = True)

    return df_out
def np_pearson_cor(X,chunk_size=1000):
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()
    
    n_genes = X.shape[1]
    corr_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
    
    gene_sums = X.sum(axis=0).A1  
    gene_means = gene_sums / X.shape[0] 
    
    for i in tqdm(range(0, n_genes, chunk_size)):
        end_i = min(i + chunk_size, n_genes)
        
        X_chunk = X[:, i:end_i].copy()
        
        for j in range(X_chunk.shape[1]):
            X_chunk.data[X_chunk.indptr[j]:X_chunk.indptr[j+1]] -= gene_means[i+j]
        
        cov_chunk = X.T.dot(X_chunk).A 
        
        std_i = np.sqrt(np.sum(X_chunk.power(2), axis=0).A1)
        std_all = np.sqrt(np.sum(X.power(2), axis=0).A1)
        
        denominator = np.outer(std_all, std_i)
        corr_chunk = cov_chunk / (X.shape[0] * denominator)
        
        corr_chunk = np.clip(corr_chunk, -1.0, 1.0)
        
        corr_matrix[:, i:end_i] = corr_chunk
    
    np.fill_diagonal(corr_matrix, 1.0)
    
    return corr_matrix

def get_coexpression_network_from_train(adata, threshold, k, data_path, data_name, split, seed, train_gene_set_size):
    
    
    path = Path(data_path) / data_name
    path.mkdir(parents=True, exist_ok=True) 

    fname = path / f"{split}_{seed}_{train_gene_set_size}_{threshold}_{k}_co_expression_network.csv"
    print(fname)
    if os.path.exists(fname):
        return pd.read_csv(fname)
    else:
        gene_list = [f for f in adata.var.index.values]
        idx2gene = dict(zip(range(len(gene_list)), gene_list)) 
    

        print("Get dense matrix")
        out = np_pearson_cor(adata.X)
        print("Finish calculation of person_cor")
        out[np.isnan(out)] = 0
        out = np.abs(out)
        rows, _ = out.shape
        out_sort_idx = np.argsort(out)[:, -21:]  # 等价于 [:, -k:]
        out_sort_val = np.sort(out)[:, -21:]

        df_g = []
        for i in range(out_sort_idx.shape[0]):
            target = idx2gene[i]
            for j in range(out_sort_idx.shape[1]):
                df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))

        df_g = [i for i in df_g if i[2] > threshold]

        df_co_expression = pd.DataFrame(df_g).rename(columns = {0: 'source', 1: 'target', 2: 'importance'})

        df_co_expression.to_csv(fname, index = False)
        print("Save to path")
        return df_co_expression
def get_go_auto(gene_list, data_path, data_name):
    go_path = os.path.join(data_path, 'go_networks.csv')
    
    if os.path.exists(go_path):
        return pd.read_csv(go_path)
    else:
        with open(os.path.join('/gene2go_all.pkl'), 'rb') as f:
            gene2go = pickle.load(f)

        gene2go = {i: list(gene2go[i]) for i in gene_list if i in gene2go}
        edge_list = []
        for g1 in tqdm(gene2go.keys()):
            for g2 in gene2go.keys():
                edge_list.append((g1, g2, len(np.intersect1d(gene2go[g1], gene2go[g2]))/len(np.union1d(gene2go[g1], gene2go[g2]))))

        further_filter = [i for i in edge_list if i[2] > 0.1]
        df_edge_list = pd.DataFrame(further_filter).rename(columns = {0: 'gene1', 1: 'gene2', 2: 'score'})

        df_edge_list = df_edge_list.rename(columns = {'gene1': 'source', 'gene2': 'target', 'score': 'importance'})

        df_edge_list.to_csv(go_path, index = False)        
        return df_edge_list