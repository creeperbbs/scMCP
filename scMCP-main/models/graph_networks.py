import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm
import pickle
import sys, os
import requests
from torch_geometric.data import Data

from zipfile import ZipFile 

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from dcor import distance_correlation, partial_distance_correlation
from sklearn.metrics import r2_score
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__(aggr='add')  
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        
        self.edge_embedding = nn.Linear(1, emb_dim)  

    def forward(self, x, edge_index, edge_weight):
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(1)
        edge_embeddings = self.edge_embedding(edge_weight)
    
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr  

    def update(self, aggr_out):
        return self.mlp(aggr_out)  

class GeneGINet(nn.Module):
    """
    Args:
        num_genes (int): num of genes
        num_layer (int): layers
        emb_dim (int): dim of embedding
        feat_dim (int): dim of features
        drop_ratio (float): dropout rate
    Output:
        latent
    """
    def __init__(self, num_genes, num_layer=2, emb_dim=300, feat_dim=256, drop_ratio=0):
        super(GeneGINet, self).__init__()
        self.num_genes = num_genes
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        
        self.x_projection = nn.Linear(1, emb_dim)
        
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))
        
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        # self.pool = global_add_pool
        
        # self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        
    def forward(self, x, edge_index, edge_weight):
        # x: [batch_size, num_genes]
        
        # reshape to adapt linear
        batch_size = x.size(0)
        h = x.view(batch_size * self.num_genes, x.shape[2])  # [batch_size*num_genes, 1]
        
        # to latent
        
        # create  batch
        batch = torch.repeat_interleave(torch.arange(batch_size, device=x.device), self.num_genes)
        
        for layer in range(self.num_layer):
            h = self.gnns[layer].to(x.device)(h.to(x.device), edge_index.to(x.device), edge_weight.to(x.device))
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        # print(h.shape)
        # # h_graph = self.pool(h, batch)  
        # print(h_graph.shape)
        # # h = self.feat_lin(h_graph)
        # print(h.shape)
        gene_embeddings = h.view(batch_size, self.num_genes, -1)  # [batch_size, num_genes, feat_dim]
        
        return gene_embeddings
