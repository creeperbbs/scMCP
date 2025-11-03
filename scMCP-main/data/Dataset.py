# -*- coding: utf-8 -*-
# @Author: Hongzhi Yao
# @Date:   2025-06-21 09:05:55
# @Last Modified by:   Hongzhi Yao
# @Last Modified time: 2025-6-21 16:33:14
import os
import scanpy as sc
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import random

import pandas as pd
import torch 
from torch.utils.data import Dataset
from ._utils import Condition_encoder, Drug_SMILES_encode, rank_genes_groups_by_cov, Drug_dose_encoder,load_smiles_bedding_drug_dose_encoder,load_smiles_from_path_embedding_drug_dose_encoder,get_drug_embeddings
import pickle
file = open('human_mouse_tokens.pickle', 'rb')
id_token = pickle.load(file)
file.close()
import functools

file = open('Gene_id_name_dict.pickle', 'rb')
gene = pickle.load(file)
file.close()
name2id = {value:key for key,value in gene.items()}
class AnnDataset(Dataset):
    '''
    Dataset for loading tensors from AnnData objects.
    ''' 
    def __init__(self,
                 adata,
                 dtype='train',
                 comb_num=1,split_key = 'random_split_0',
                 obs_key ='cell_type'
                 ):

        self.dense_adata = adata

        if sparse.issparse(adata.X):
            self.dense_adata  = sc.AnnData(X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True))
        
        self.dense_data = torch.tensor(self.dense_adata.X, dtype=torch.float32)
        da_obs = self.dense_adata.obs[[obs_key]].copy()
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        obs_encoded = encoder.fit_transform(da_obs)
        self.cell_type = torch.tensor(obs_encoded,dtype=torch.float32)
        self.cell_name=self.dense_adata.obs[obs_key].to_list()
    def __len__(self):
        return len(self.dense_data)

    def __getitem__(self, index):
        return {'data':self.dense_data[index,:],'obs':self.cell_type[index,:],'celltype': self.cell_name[index]}
from sklearn.preprocessing import LabelEncoder
class DrugDoseAnnTokenDataset(Dataset):

    def __init__(self,
                 adata,
                 dtype='train',
                 obs_key='cov_drug',
                 comb_num=1,
                 split_key='random_split_0',
                 top_gene_per_cell=2048 ,
                  smiles_dataset='sciplex'

                 ):
        self.dtype = dtype
        self.obs_key = obs_key        
        self.top_gene_per_cell = top_gene_per_cell 

        if sparse.issparse(adata.X):
            self.dense_adata = sc.AnnData(
                X=adata.X.A, 
                obs=adata.obs.copy(deep=True),
                var=adata.var.copy(deep=True),
                uns=adata.uns.copy(deep=True)
            )
        else:
            self.dense_adata = adata

        self.all_gene_ids = self.dense_adata.var.index.tolist() 
        self.total_gene_num = len(self.all_gene_ids)  

        if self.total_gene_num < self.top_gene_per_cell:
            self.top_gene_per_cell = self.total_gene_num

        self.drug_adata = self.dense_adata[self.dense_adata.obs['dose'] != 0.0]
        self.data = torch.tensor(self.drug_adata.X, dtype=torch.float32)
        self.dense_data = torch.tensor(self.dense_adata.X, dtype=torch.float32)
        
        self.paired_control_index = self.drug_adata.obs['paired_control_index'].tolist()
        self.dense_adata_index = self.dense_adata.obs.index.to_list()  
        self.drug_type_list = self.drug_adata.obs['SMILES'].to_list()
        self.dose_list = self.drug_adata.obs['dose'].to_list()
        self.obs_list = self.drug_adata.obs[obs_key].to_list()
        

        if smiles_dataset!='sciplex' and smiles_dataset!='tagoe':
            
            self.encode_drug_doses = get_drug_embeddings(drug_SMILES_list=self.drug_type_list,dose_list=self.dose_list)
        else:
            self.encode_drug_doses = load_smiles_bedding_drug_dose_encoder(drug_SMILES_list=self.drug_type_list,dataset=smiles_dataset, dose_list=self.dose_list)
        self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)
    def __len__(self):
        return len(self.drug_adata)  


    def __getitem__(self, index):
        outputs = dict()
        control_idx = self.dense_adata_index.index(str(self.paired_control_index[index]))
        control_expr = self.dense_data[control_idx, :]  
        top_vals, top_indices = torch.topk(control_expr, self.top_gene_per_cell, dim=0)
        outputs['x_top_vals'] = self.data[index]  
        outputs['x_top_indices'] = top_indices 
        outputs['control_top_vals'] = top_vals
        top_gene_tokens = []
        top_gene_ids = [] 
        for idx in top_indices:
            gene_id = self.all_gene_ids[idx.item()]  
            top_gene_ids.append(gene_id)
            top_gene_tokens.append(id_token.get(name2id.get(gene_id), 0)) 

        outputs['top_gene_tokens'] = torch.tensor(top_gene_tokens, dtype=torch.long)
        outputs['top_gene_ids'] = top_gene_ids 

        outputs['drug_dose'] = self.encode_drug_doses[index, :]
        outputs['label'] = outputs['drug_dose']
        outputs['cov_drug'] = self.obs_list[index]
        outputs['data'] = self.dense_adata[index, :] 
        
        return {
            'features': (outputs['control_top_vals'], outputs['x_top_vals']),
            'label': outputs['label'],
             # 'batch':(self.batch[control_idx],self.batch[index]),
            'cov_drug': outputs['cov_drug'],
            'top_gene_tokens': outputs['top_gene_tokens'],
            'top_gene_indices': outputs['x_top_indices'],  
            'top_gene_ids': outputs['top_gene_ids'],
            'control_raw': control_expr
            # 'celltype':self.celltype_list[index]
        }
class EvalAnnDataset(Dataset):
    '''
    Dataset for loading tensors from AnnData objects.
    ''' 
    def __init__(self,
                 adata,
                 dtype='train',
                 comb_num=1,split_key = 'random_split_0',
                 smiles_dataset = None
                 ):

        self.dense_adata = adata

        if sparse.issparse(adata.X):
            self.dense_adata  = sc.AnnData(X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True))
        
        self.dense_data = torch.tensor(self.dense_adata.X, dtype=torch.float32)
        self.drug_adata = self.dense_adata[self.dense_adata.obs['dose']!=0.0]
        self.dose_list = self.drug_adata.obs['dose'].to_list()
        self.drug_type_list = self.drug_adata.obs['SMILES'].to_list()
        self.encode_drug_doses = load_smiles_from_path_embedding_drug_dose_encoder(drug_SMILES_list=self.drug_type_list,file_path=smiles_dataset, dose_list=self.dose_list)
        self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)
    def __len__(self):
        return len(self.dense_data)

    def __getitem__(self, index):
        outputs = dict()
        outputs['drug_dose'] = self.encode_drug_doses[index, :]
        return {'data':self.dense_data[index,:],'label':outputs['drug_dose']}
class DrugDoseAnnDataset(Dataset):
    '''
    Dataset for loading tensors from AnnData objects.
    ''' 
    def __init__(self,
                 adata,
                 dtype='train',
                 obs_key='cov_drug',
                 smiles_dataset='sciplex',
                 comb_num=1,split_key = 'random_split_0'
                 ):
        self.dtype = dtype
        self.obs_key = obs_key        
        
        
        self.dense_adata = adata

        if sparse.issparse(adata.X):
            self.dense_adata  = sc.AnnData(X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True))
        
        self.drug_adata = self.dense_adata[self.dense_adata.obs['dose']!=0.0] 
        self.data = torch.tensor(self.drug_adata.X, dtype=torch.float32)
        self.dense_data = torch.tensor(self.dense_adata.X, dtype=torch.float32)

        self.paired_control_index = self.drug_adata.obs['paired_control_index'].tolist()
        self.dense_adata_index = self.dense_adata.obs.index.to_list()


        self.drug_type_list = self.drug_adata.obs['SMILES'].to_list()
        self.dose_list = self.drug_adata.obs['dose'].to_list()

        da_obs = self.dense_adata.obs[['cell_type']].copy()
        self.cell_name=self.dense_adata.obs['cell_type'].to_list()
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        obs_encoded = encoder.fit_transform(da_obs)
#         self.batch_name=self.dense_adata.obs[['batch']].copy()
#         batch_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
#         batch_encoded = batch_encoder.fit_transform(self.batch_name)
        
        
        from sklearn.preprocessing import LabelEncoder

        self.batch_name = self.dense_adata.obs['batch'].values  # 原始字符串
        batch_encoder = LabelEncoder()          # 0,1,2,...
        self.batch = batch_encoder.fit_transform(self.batch_name) 
        self.batch = torch.tensor(self.batch,dtype=torch.float32)
        
        self.cell_type = torch.tensor(obs_encoded,dtype=torch.float32)
        self.var_list = self.drug_adata.var.index.to_list()
        # self.obs_list = self.drug_adata.obs[obs_key].to_list()
        # file_name = f'./embedding/1024dPRNet_drug_emb_{split_key}_DrugGCL_{dtype}.npy'
        
        if smiles_dataset!='sciplex' and smiles_dataset!='tagoe':
            self.encode_drug_doses = load_smiles_from_path_embedding_drug_dose_encoder(drug_SMILES_list=self.drug_type_list,file_path=smiles_dataset, dose_list=self.dose_list)
        else:
            self.encode_drug_doses = load_smiles_bedding_drug_dose_encoder(drug_SMILES_list=self.drug_type_list,dataset=smiles_dataset, dose_list=self.dose_list)
        self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)



    def __len__(self):
        return len(self.drug_adata)

    def __getitem__(self, index):
        outputs = dict()
        outputs['x'] = self.data[index, :]
        control_index = self.dense_adata_index.index(str(self.paired_control_index[index])) 
            
        outputs['control'] = self.dense_data[control_index,:]
        outputs['drug_dose'] = self.encode_drug_doses[index, :]
        outputs['label'] = outputs['drug_dose']

        # obs_info = self.obs_list[index]
        outputs['obs'] = self.cell_type[index]
        # outputs['cov_drug'] = obs_info
        # var_info = self.var_list[index]
        
        return {'features':(outputs['control'], outputs['x']), 'batch':(self.batch[control_index, :],self.batch[index, :]),'label':outputs['label'],'data':self.dense_data[index,:],'obs':outputs['obs'],'celltype': self.cell_name[index]}
    
    
