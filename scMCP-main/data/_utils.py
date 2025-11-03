# -*- coding: utf-8 -*-
# @Author: Hongzhi Yao
# @Date:   2025-06-21 09:05:55
# @Last Modified by:   Hongzhi Yao
# @Last Modified time: 2025-6-21 16:33:14
import scanpy as sc
import numpy as np
import torch.nn.functional as F
from argparse import Namespace
import numpy as np
import random
from collections import OrderedDict
import pandas as pd
import anndata as ad
from functools import partial
# import pytorch_lightning as pl
import yaml

from torch_geometric.data import Data, Dataset, DataLoader
import pickle
import sys
sys.path.append('/home/MBDAI206AA201/jupyter/yhz/sc/scdata/GeneCompass-main/downstream_tasks/PRCEdrug/GraphEmbedding/DrugGCL')
from ginet_3emb_degree import GNNDecoder, GNN,GNN_fp
from scipy import sparse
from scipy.stats import wasserstein_distance
from rdkit import Chem
from rdkit.Chem import AllChem

from loss import NTXentLoss, sce_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing
from tqdm import tqdm  


from torch_geometric.data import Batch
def bool_to_int(value):
    return 1 if value else 0

def batch_split(data, batch_size=300):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size

def load_smiles_from_path_embedding_drug_dose_encoder(drug_SMILES_list: list, dose_list: list,file_path = None, num_Bits=768, comb_num=1,batch_size =1024):
    
    drug_len = len(drug_SMILES_list)
    if drug_len == 0:
        return np.zeros((0, 0))
    
    if file_path != None:
        with open(file_path, 'rb') as f:
                embeddings_dict = pickle.load(f)    
    drug_embeddings = np.zeros((drug_len, 768))
    if comb_num == 1:
        for start_idx in tqdm(range(0, drug_len, batch_size)):
            end_idx = min(start_idx + batch_size, drug_len)
            batch_C = [embeddings_dict[smiles] for smiles in drug_SMILES_list[start_idx:end_idx]]
            batch_C = np.array(batch_C)
            batch_doses = np.array(dose_list[start_idx:end_idx])
            scaling_factors = np.log10(batch_doses + 1).reshape(-1, 1)
            drug_embeddings[start_idx:end_idx] = batch_C * scaling_factors
    else:
        all_sub_smiles = []
        group_indices = []
        
        for i, smiles in enumerate(drug_SMILES_list):
            subs = smiles.split('+')
            all_sub_smiles.extend(subs)
            group_indices.extend([i] * len(subs))
        sub_embeddings = []
        for start_idx in range(0, len(all_sub_smiles), batch_size):
            end_idx = min(start_idx + batch_size, len(all_sub_smiles))
            batch_smiles = all_sub_smiles[start_idx:end_idx]
            
            batch_C = embed(lm.encoder, batch_smiles, tokenizer,128)
            sub_embeddings.append(batch_C)
        
        sub_embeddings = np.concatenate(sub_embeddings, axis=0)
        
        group_indices = np.array(group_indices)
        for i in range(drug_len):
            mask = group_indices == i
            if mask.any():
                drug_embeddings[i] = sub_embeddings[mask].sum(axis=0)
        
        scaling_factors = np.log10(np.array(dose_list) + 1).reshape(-1, 1)
        drug_embeddings *= scaling_factors

    return drug_embeddings
def load_smiles_bedding_drug_dose_encoder(drug_SMILES_list: list, dose_list: list,dataset='sciplex', num_Bits=768, comb_num=1,batch_size =1024):
    
    drug_len = len(drug_SMILES_list)
    if drug_len == 0:
        return np.zeros((0, 0))
    
    if dataset == 'sciplex':
        file_path = '/embeddings/sciplex_smiles_embeddings.pkl'
    elif dataset == 'tagoe':
        file_path = '/embeddings/Tagoe_smiles_embeddings.pkl'
    with open(file_path, 'rb') as f:
            embeddings_dict = pickle.load(f)    
    drug_embeddings = np.zeros((drug_len, 768))
    print(drug_len)
    if comb_num == 1:
        for start_idx in tqdm(range(0, drug_len, batch_size)):
            end_idx = min(start_idx + batch_size, drug_len)
            batch_C = [embeddings_dict[smiles] for smiles in drug_SMILES_list[start_idx:end_idx]]
            batch_C = np.array(batch_C)
            batch_doses = np.array(dose_list[start_idx:end_idx])
            scaling_factors = np.log10(batch_doses + 1).reshape(-1, 1)
            drug_embeddings[start_idx:end_idx] = batch_C * scaling_factors
    else:
        all_sub_smiles = []
        group_indices = []
        
        for i, smiles in enumerate(drug_SMILES_list):
            subs = smiles.split('+')
            all_sub_smiles.extend(subs)
            group_indices.extend([i] * len(subs))
        sub_embeddings = []
        for start_idx in range(0, len(all_sub_smiles), batch_size):
            end_idx = min(start_idx + batch_size, len(all_sub_smiles))
            batch_smiles = all_sub_smiles[start_idx:end_idx]
            
            batch_C = embed(lm.encoder, batch_smiles, tokenizer,128)
            sub_embeddings.append(batch_C)
        
        sub_embeddings = np.concatenate(sub_embeddings, axis=0)
        
        group_indices = np.array(group_indices)
        for i in range(drug_len):
            mask = group_indices == i
            if mask.any():
                drug_embeddings[i] = sub_embeddings[mask].sum(axis=0)
        
        scaling_factors = np.log10(np.array(dose_list) + 1).reshape(-1, 1)
        drug_embeddings *= scaling_factors

    return drug_embeddings

def get_drug_embeddings(drug_SMILES_list, dose_list, comb_num=1, dataset='sciplex', batch_size=64):
    drug_len = len(drug_SMILES_list)
    if drug_len == 0:
        return np.zeros((0, 0))
    
    if dataset == 'sciplex':
        file_path = '/embeddings/sciplex_smiles_embeddings.pkl'
    elif dataset == 'tagoe':
        file_path = '/embeddings/Tagoe_smiles_embeddings.pkl'
    else:
        file_path = None
    
    file_path_fda = '/embeddings/FDA_smiles_embeddings.pkl'
    
    with open(file_path_fda, 'rb') as f:
        fda_embeddings = pickle.load(f)
    
    primary_embeddings = {}
    if file_path:
        with open(file_path, 'rb') as f:
            primary_embeddings = pickle.load(f)
    
    drug_embeddings = np.zeros((drug_len, 768))
    
    if comb_num == 1:
        for start_idx in range(0, drug_len, batch_size):
            end_idx = min(start_idx + batch_size, drug_len)
            batch_C = []
            for smiles in drug_SMILES_list[start_idx:end_idx]:
                if smiles in primary_embeddings:
                    emb = primary_embeddings[smiles]
                elif smiles in fda_embeddings:
                    emb = fda_embeddings[smiles]
                else:
                    emb = np.zeros(768)
                batch_C.append(emb)
            batch_C = np.array(batch_C)
            scaling_factors = np.log10(np.array(dose_list[start_idx:end_idx]) + 1).reshape(-1, 1)
            drug_embeddings[start_idx:end_idx] = batch_C * scaling_factors
    else:
        all_sub_smiles = []
        group_indices = []
        for i, smiles in enumerate(drug_SMILES_list):
            subs = smiles.split('+')
            all_sub_smiles.extend(subs)
            group_indices.extend([i] * len(subs))
        
        sub_embeddings = []
        for start_idx in range(0, len(all_sub_smiles), batch_size):
            end_idx = min(start_idx + batch_size, len(all_sub_smiles))
            batch_smiles = all_sub_smiles[start_idx:end_idx]
            batch_emb = []
            for s in batch_smiles:
                if s in primary_embeddings:
                    emb = primary_embeddings[s]
                elif s in fda_embeddings:
                    emb = fda_embeddings[s]
                else:
                    emb = np.zeros(768)
                batch_emb.append(emb)
            sub_embeddings.extend(batch_emb)
        
        sub_embeddings = np.array(sub_embeddings)
        group_indices = np.array(group_indices)
        
        for i in range(drug_len):
            mask = group_indices == i
            drug_embeddings[i] = sub_embeddings[mask].sum(axis=0)
        
        scaling_factors = np.log10(np.array(dose_list) + 1).reshape(-1, 1)
        drug_embeddings *= scaling_factors
    
    return drug_embeddings


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=2,
    rankby_abs=True,
    key_added='rank_genes_groups_cov',
    return_dict=False,
):

    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    print(cov_categories)
    for cov_cat in cov_categories:
        #name of the control group in the groupby obs column
        control_group_cov = '_'.join([cov_cat, control_group])

        #subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate]==cov_cat]

        #compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False
        )

        #add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns['rank_genes_groups']['names'])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict




def rank_genes_groups_by_drug(
    adata,
    groupby,
    control_group,
    pool_doses=False,
    n_genes=2,
    rankby_abs=True,
    key_added='rank_genes_groups_drug',
    return_dict=False,
):

    gene_dict = {}
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        reference=control_group,
        rankby_abs=rankby_abs,
        n_genes=n_genes,
        use_raw=False
    )

    #add entries to dictionary of gene sets
    de_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    for group in de_genes:
        gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict




def pearson_mean(data1, data2):
    sum_pearson_1 = 0
    sum_pearson_2 = 0
    for i in range(data1.shape[0]):
        pearsonr_ = pearsonr(data1[i], data2[i])
        sum_pearson_1 += pearsonr_[0]
        sum_pearson_2 += pearsonr_[1]
    return sum_pearson_1/data1.shape[0], sum_pearson_2/data1.shape[0]

def pearson_list(data1, data2):
    pearson_list = np.zeros(data1.shape[0])
    for i in range(data1.shape[0]):
        pearsonr_ = pearsonr(data1[i], data2[i])
        pearson_list[i] = pearsonr_[0]
    return pearson_list


def r2_mean(data1, data2):
    sum_r2_1 = 0
    for i in range(data1.shape[0]):
        r2_score_ = r2_score(data1[i], data2[i])
        sum_r2_1 += r2_score_           
    return sum_r2_1/data1.shape[0]

def mse_mean(data1, data2):
    sum_mse_1 = 0
    for i in range(data1.shape[0]):
        mse_score_ = mean_squared_error(data1[i], data2[i])
        sum_mse_1 += mse_score_           
    return sum_mse_1/data1.shape[0]

def z_score(control_array, drug_array):
    scaler = preprocessing.StandardScaler()
    array_all = np.concatenate((control_array, drug_array),axis=0)
    scaler.fit(array_all)
    control_array_z = scaler.transform(control_array)
    drug_array_z = scaler.transform(drug_array)
    return control_array_z, drug_array_z

def contribution_df(data):
    data['cov_drug_name'] = data.index
    data['cell_type'] = data.cov_drug_name.apply(lambda x: str(x).split('_')[0])
    data['condition'] = data.cov_drug_name.apply(lambda x: '_'.join(str(x).split('_')[1:]))
    return data



def condition_fc_groups_by_cov(
    data,
    groupby,
    control_group,
    covariate
):

    condition_exp_mean = {}
    control_exp_mean = {}
    fold_change = {}

    cov_categories = data[covariate].unique()
    for cov_cat in cov_categories:
        #name of the control group in the groupby obs column
        control_group_cov = '_'.join([str(cov_cat), str(control_group)])

        #subset adata to cells belonging to a covariate category
        adata_cov_df = data[data[covariate]==cov_cat]
        adata_cov_df['condition'] = data[groupby]


        control_mean = adata_cov_df[adata_cov_df.cov_drug_name == control_group_cov].mean(numeric_only=True)
        control_exp_mean[control_group_cov] = control_mean

        for cond, df in tqdm(adata_cov_df.groupby('condition')): 
            if df.shape[0] != 0 :
                if cond != control_group_cov:
                    drug_mean = df.mean(numeric_only=True)
                    fold_change[cond] = drug_mean-control_mean
                    condition_exp_mean[cond] = drug_mean

    return condition_exp_mean, control_exp_mean, fold_change


    

# This file consists of useful functions that are related to cmap. Reference: https://github.com/kekegg/DLEPS
def computecs(qup, qdown, expression):
    '''
    This function takes qup & qdown, which are lists of gene
    names, and  expression, a panda data frame of the expressions
    of genes as input, and output the connectivity score vector
    '''
    r1 = ranklist(expression)
    if qup and qdown:
        esup = computees(qup, r1)
        esdown = computees(qdown, r1)
        w = []
        for i in tqdm(range(len(esup))):
            if esup[i]*esdown[i] <= 0:
                w.append(esup[i]-esdown[i])
            else:
                w.append(0)
        return pd.DataFrame(w, expression.columns)
    elif qup and qdown==None:
        esup = computees(qup, r1)
        return pd.DataFrame(esup, expression.columns)
    elif qup == None and qdown:
        esdown = computees(qdown, r1)
        return pd.DataFrame(esdown, expression.columns)
    else:
        return None

def computees(q, r1):
    '''
    This function takes q, a list of gene names, and r1, a panda data
    frame as the input, and output the enrichment score vector
    '''
    if len(q) == 0:
        ks = 0
    elif len(q) == 1:
        ks = r1.loc[q,:]
        ks.index = [0]
        ks = ks.T
#print(ks)
    else:
        n = r1.shape[0]
        sub = r1.loc[q,:]
        J = sub.rank()
        a_vect = J/len(q)-sub/n
        b_vect = (sub-1)/n-(J-1)/len(q)
        a = a_vect.max()
        b = b_vect.max()
        ks = []
        for i in range(len(a)):
            if a[i] > b[i]:
                ks.append(a[i])
            else:
                ks.append(-b[i])
#print(ks)
    return ks
def ranklist(DT):
    # This function takes a panda data frame of gene names and expressions
    # as an input, and output a data frame of gene names and ranks
    ranks = DT.rank(ascending=False, method="first")
    return ranks

    












