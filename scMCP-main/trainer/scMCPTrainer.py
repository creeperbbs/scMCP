# -*- coding: utf-8 -*-
# @Author: Hongzhi Yao
# @Date:   2025-06-21 09:05:55
# @Last Modified by:   Hongzhi Yao
# @Last Modified time: 2025-4-21 16:33:14
import os
import networkx as nx
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn import metrics
from scipy.spatial import distance_matrix, minkowski_distance, distance
from torch.distributions import Laplace
import torch
import torch.nn as nn
import itertools
from torch.autograd import Variable, grad
from functools import partial
from torch.nn import functional as F
from torch.distributions import NegativeBinomial, normal
from torch.cuda.amp import GradScaler
from itertools import cycle
from itertools import islice
from torch.distributions import Bernoulli, Normal

import math
from tqdm import tqdm
from igraph import *
import umap
import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error
from torch import autocast
from sklearn.metrics.pairwise import cosine_similarity
from data.Dataset import  AnnDataset
from data.Dataset import  DrugDoseAnnDataset,DrugDoseAnnTokenDataset
from models.scMCP import scMCP
from models.DA import Discriminator, GeneratorDA, GeneratorAD
from trainer.loss import mmd_loss
# from models.PRnet import PRnet
from ._utils import train_valid_test,train_valid_test_no_dose
from .go_networks import get_similarity_network,GeneSimNetwork

def dist_loss(x,y,GAMMA=10):
        result = mmd_loss(x,y,GAMMA)
        return result
def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)
from torch.optim.lr_scheduler import _LRScheduler

def calculateKNNgraphDistanceMatrix(featureMatrix, distanceType='euclidean', k=10):
    distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
        #print(distMat)
    edgeList=[]
    
    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k):
            edgeList.append((i,res[j],distMat[i,j]))
    
    return edgeList
def generateLouvainCluster(edgeList):
    """
    Louvain Clustering using igraph
    """
    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeList)
    W = nx.adjacency_matrix(Gtmp).toarray()
    graph = Graph.Weighted_Adjacency(
        W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False)
    louvain_partition = graph.community_multilevel(
        weights=graph.es['weight'], return_levels=False)
    size = len(louvain_partition)
    hdict = {}
    count = 0
    for i in range(size):
        tlist = louvain_partition[i]
        for j in range(len(tlist)):
            hdict[tlist[j]] = i
            count += 1

    listResult = []
    for i in range(count):
        listResult.append(hdict[i])

    return listResult, size
def calculateUMAPgraphDistanceMatrix(featureMatrix, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    if isinstance(featureMatrix, torch.Tensor):
        featureMatrix = featureMatrix.cpu().detach().numpy()
    
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2, 
        transform_mode='graph'
    )
    
    umap_graph = umap_model.fit_transform(featureMatrix)
    
    if sparse.issparse(umap_graph):
        coo_graph = umap_graph.tocoo()
        edgeList = list(zip(coo_graph.row, coo_graph.col, coo_graph.data))
    else:
        edgeList = []
        for i in range(umap_graph.shape[0]):
            for j in range(i+1, umap_graph.shape[1]):
                if umap_graph[i, j] > 0:
                    edgeList.append((i, j, umap_graph[i, j]))
                    edgeList.append((j, i, umap_graph[i, j]))  
    
    return edgeList

def direction_consistency_loss(y_true,y_pred, dir_lambda):

    sign_true = torch.sign(y_true)
    sign_pred = torch.sign(y_pred)
    return  torch.sum(dir_lambda *(sign_true-sign_pred)**2)/y_true.shape[0]/y_true.shape[1]
def _direction_consistency_loss_numpy(y_true, y_pred):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    
    mask = np.abs(y_true) > epsilon
    
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    direction_agreement = sign_true * sign_pred
    
    abs_diff = np.square(y_pred - y_true)
    
    base_loss = np.where(
        direction_agreement < 0,             
        abs_diff,                            
        np.maximum(abs_diff - tau, 0.0)     
    )
    
    masked_loss = base_loss * mask
    
    valid_count = np.sum(mask)
    if valid_count > 0:
        return np.sum(masked_loss) / valid_count
    else:
        return 0.0 
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss
from models.PRnet import PRnet
from torch.distributions import Categorical, Normal, MixtureSameFamily, Independent

def gmm_loss(gene_means, gene_vars, alphas, target):
    components_distribution = Independent(
        Normal(loc=gene_means, scale=torch.sqrt(gene_vars)), 
        reinterpreted_batch_ndims=1  
    )
    mixture = MixtureSameFamily(mixture_distribution=Categorical(probs=alphas), component_distribution=components_distribution)
    
    loss = -mixture.log_prob(target)
    return loss.mean()
with open('/home/MBDAI206AA201/jupyter/yhz/sc/MOMDGDP-main/embeddings/Lung_cancer_numbers.txt', 'r', encoding='utf-8') as f:
    index_raw = [int(line.strip()) for line in f]  # strip() 移除换行符
    
    index_raw = np.array(index_raw, dtype=int) 

class ZILNCriterion(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps 
    def forward(self, input_means, input_vars, input_zi_logits, target):
        zi_prob = torch.sigmoid(input_zi_logits)  # [batch, genes]
        
        zero_mask = (target == 0).float()  
        zi_loss = F.binary_cross_entropy_with_logits(
            input=input_zi_logits,
            target=zero_mask,
            reduction="none"
        )  # [batch, genes]

        non_zero_mask = (target > 0).float()  
        if non_zero_mask.sum() == 0:
            log_normal_loss = torch.zeros_like(zi_loss)
        else:
            input_vars = torch.clamp(input_vars, min=1e-6) 
            gaussian_nll = 0.5 * (torch.log(input_vars) + (input_means - target)** 2 / input_vars)

            log_normal_loss = gaussian_nll * non_zero_mask  

        total_loss = zi_loss + log_normal_loss
        
        return total_loss.mean()

def sample_ziln(zi_prob, log_normal_dist, device=None, truncate_max=None):

    if device is None:
        device = zi_prob.device
    bernoulli_dist = Bernoulli(probs=zi_prob)
    zero_mask = bernoulli_dist.sample().to(device)  
    
    y_samples = log_normal_dist.sample()  
    if truncate_max is not None:
        y_samples = torch.clamp(y_samples, max=truncate_max)
    

    samples = y_samples * (1 - zero_mask) 
    return samples
class scMCPTrainer:
    """
    This class contains the implementation of the MOMDGDPTrainer Trainer
    Parameters
    ----------
    model: MOMDGDP
    adata: : `~anndata.AnnData`
        Annotated Data Matrix for training MOMDGDP.
    batch_size: integer
        size of each batch to be fed to network.
    comb_num: int
        Number of combined compounds.
    shuffle: bool
        if `True` shuffles the training dataset.
    split_key: string
        Attributes of data split.
    model_save_dir: string
        Save dir of model. 
    x_dimension: int
        Dimention of x
    hidden_layer_sizes: list
        A list of hidden layer sizes
    z_dimension: int
        Dimention of latent space
    adaptor_layer_sizes: list
        A list of adaptor layer sizes
    comb_dimension: int
        Dimention of perturbation latent space
    drug_dimension: int
        Dimention of rFCGP
    n_genes: int
        Dimention of different expressed gene
    n_epochs: int
        Number of epochs to iterate and optimize network weights.
    train_frac: Float
        Defines the fraction of data that is used for training and data that is used for validation.
    dr_rate: float
        dropout_rate
    loss: list
        Loss of model, subset of 'NB', 'GUSS', 'KL', 'MSE'
    obs_key:
        observation key of data
    """
    def __init__(self, adata,target_adata=None, batch_size = 32, comb_num = 2, shuffle = True, split_key='random_split', model_save_dir = './checkpoint/',results_save_dir = './results/', x_dimension = 5000, hidden_layer_sizes = [128], z_dimension = 64, adaptor_layer_sizes = [128], comb_dimension = 64, drug_dimension = 1031, n_genes=20,  dr_rate = 0.05, loss = ['GUSS'],dataset='tageo',pos_emb_graph=None,gene_emb_adapt=False, obs_key = 'cov_drug_name',pretrain_vae=False, da_mode=False,**kwargs): # maybe add more parameters
        
        assert set(loss).issubset(['NB', 'GUSS', 'KL', 'MSE','ZINB']), "loss should be subset of ['NB', 'GUSS', 'KL', 'MSE']"

        self.x_dim = x_dimension
        self.split_key = split_key
        self.z_dimension = z_dimension
        self.comb_dimension = comb_dimension
        self.da_mode = da_mode
        self.model_save_dir = model_save_dir
        self.results_save_dir = results_save_dir
        self.loss = loss
        self.pretrain_vae = pretrain_vae

        self.seed = kwargs.get("seed", 2025)
        torch.manual_seed(self.seed)
        # self.model.apply(self.weight_init)
        self.adata = adata
        #self.adata_deg_list = adata.uns['rank_genes_groups_cov']
        self.de_n_genes = n_genes
        self.adata_var_names = adata.var_names
        self.train_data, self.valid_data, self.test_data = train_valid_test(self.adata, split_key = split_key)
        self.sim_network = None
        if pos_emb_graph == 'co_expression':
            ## calculating co expression similarity graph
            coexpress_threshold = 0.1
            num_similar_genes_co_express_graph = 20,
            self.gene_list = list(adata.var.index)
            self.data_path = '/embeddings/'
            self.dataset_name = dataset
            self.split = split_key
            self.train_gene_set_size = x_dimension
            self.node_map = {x: it for it, x in enumerate(self.gene_list)}
            edge_list = get_similarity_network(network_type = 'co-express', adata = adata, 
                                               threshold = coexpress_threshold, 
                                               k = num_similar_genes_co_express_graph, gene_list = self.gene_list, data_path = self.data_path, data_name = self.dataset_name, split = self.split, seed =self.seed, train_gene_set_size = self.train_gene_set_size)
            
            self.sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
        elif pos_emb_graph == 'go':
                    ## calculating co expression similarity graph
                    coexpress_threshold = 0.1
                    num_similar_genes_co_express_graph = 20,
                    self.gene_list = list(adata.var.index)
                    self.data_path = '/embeddings/'
                    self.dataset_name = dataset
                    self.split = split_key
                    self.train_gene_set_size = x_dimension
                    self.node_map = {x: it for it, x in enumerate(self.gene_list)}
                    edge_list = get_similarity_network(network_type = pos_emb_graph, adata = adata, 
                                                       threshold = coexpress_threshold, 
                                                       k = num_similar_genes_co_express_graph, gene_list = self.gene_list, data_path = self.data_path, data_name = self.dataset_name, split = self.split, seed =self.seed, train_gene_set_size = self.train_gene_set_size)
                    self.sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
              self.model = scMCP(
                num_genes=2048,
                     uncertainty=False,
                     num_gnn_layers=None,
                     decoder_hidden_size=None,
                     num_gene_gnn_layers=None,
                     input_genes_ens_ids=None,
            gene_mask = index_raw,
                     scfm_genes_ens_ids=None,
            da_mode = False,
            gene_emb_adapt=gene_emb_adapt,
                coexpress_network = self.sim_network,
                drug_dim=1024,
                     hidden_size = 128,
                pos_emb_graph = pos_emb_graph,
                     grn_node2vec_file='/emb_grn/grn_emb_total.pkl',
                     ppi_node2vec_file='/emb_ppi/ppi_emb_total.pkl',
                     model_type = 'ppi_grn_mode')
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            if(torch.cuda.device_count() > 1):
                self.model = nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())])         

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        


        self.model.apply(self.weight_init)
        if self.train_data is not None:
            self.train_dataset = DrugDoseAnnTokenDataset(self.train_data, dtype='train', obs_key=obs_key,smiles_dataset=dataset,  comb_num=comb_num,split_key=split_key)     
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,shuffle=True,num_workers=10)
        if self.valid_data is not None:
            self.valid_dataset = DrugDoseAnnTokenDataset(self.valid_data, dtype='valid', obs_key=obs_key, smiles_dataset=dataset, comb_num=comb_num,split_key=split_key)
            self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size,shuffle=True,num_workers=10)
        if self.test_data is not None:
            self.test_dataset = DrugDoseAnnTokenDataset(self.test_data, dtype='test', obs_key=obs_key, comb_num=comb_num,smiles_dataset=dataset,split_key=split_key)
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True,num_workers=10)
        if self.da_mode == True:
            self._init_model(1)
            if target_adata is not None:
                # self.target_train_data, self.target_valid_data, self.target_test_data = train_valid_test(target_adata, split_key = split_key)
                self.target_train_data, self.target_valid_data, self.target_test_data = train_valid_test_no_dose(target_adata, split_key = 'random_split0')
        # train_valid_test
                self.target_train_dataset =  AnnDataset(self.target_train_data,obs_key='Subclass_Cell_Identity')
                self.target_train_dataloader = torch.utils.data.DataLoader(self.target_train_dataset, batch_size=batch_size, shuffle=True)
                # self.target_train_dataset = DrugDoseAnnDataset(self.target_train_data, dtype='train', obs_key=obs_key,smiles_dataset='tagoe',  comb_num=comb_num,split_key=split_key)
                # self.target_train_dataloader = torch.utils.data.DataLoader(self.target_train_dataset, batch_size=batch_size,shuffle=True)
                if self.target_test_data is not None:
                    self.target_test_dataset =  AnnDataset(self.target_test_data,obs_key='Subclass_Cell_Identity')
                    # self.target_test_dataset = DrugDoseAnnDataset(self.target_test_data, dtype='test', obs_key=obs_key, comb_num=comb_num,smiles_dataset='tagoe',split_key=split_key)
                    self.target_test_dataloader = torch.utils.data.DataLoader(self.target_test_dataset, batch_size=batch_size, shuffle=True)
                    # self.target_test_dataloader = torch.utils.data.DataLoader(self.target_test_dataset, batch_size=batch_size, shuffle=True)

        if set(['ZINLL']).issubset(loss):
            self.criterion =ZILNCriterion(eps=1e-8)
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.shuffle = shuffle
        self.batch_size = batch_size
        # Optimization attributes

        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        self.best_state_dictPGM = None


        self.PGM_losses = []
        self.r2_score_mean = []
        self.r2_score_var = []
        self.score = []
        self.r2_score_mean_de = []
        self.r2_score_var_de = []
        self.mse_score_de = []
        self.best_score = np.inf
        self.patient = 0
        self.sce_loss = partial(sce_loss, alpha=1) 

    def train(self, n_epochs = 100, lr = 0.001, weight_decay= 1e-8,da_mode=False, scheduler_factor=0.5,scheduler_patience=10,model_path=None,**extras_kwargs):
        
        self.n_epochs = n_epochs
        if self.da_mode is not False:
                try:
                    self.generator = GeneratorAD(
                        in_dim=17911,         
                        hidden_dim=[512, 256], 
                        num_blocks=2,         
                        mem_dim=512,       
                        threshold=0.01      
                    )
                    
                    self.generator.load_state_dict(torch.load('./_weight_modelVAE/sciplex_tagoe_724_DAE_generator_best.pkl')['model_state_dict'])
                    train_flag = False
                except Exception as e: 
                    print(f"fail to load and init model：{str(e)}")
                    train_flag = True
                if train_flag == True:
                    self.generator = GeneratorAD(
                       in_dim=17911,         
                        hidden_dim=[512, 256], 
                        num_blocks=2,         
                        mem_dim=512,       
                        threshold=0.01  ,
                        temperature = 0.05
                    ).to(self.device)
                    best_val_loss = float('inf')
                    criterion = nn.MSELoss()
                    self.dec = torch.nn.Sequential(
                        nn.Linear(17911, 256),
                        nn.ReLU(),
                         nn.LayerNorm(256),
                        nn.Linear(256, 50)).to(self.device)
                    optimizer = torch.optim.Adam( self.generator.parameters(), lr=1e-4, weight_decay= weight_decay)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.7,verbose=1,min_lr=1e-8,patience=1)
                    for epoch in range(100):
                        self.generator.train()
                        total_loss = 0
                        loop = tqdm(enumerate( self.train_dataloader), total =len(self.train_dataloader))
                        for i, data in loop:
                            control = data['data']
                            c_target = data['obs']
                            control = control.to(self.device) 
                            xc_recon, z = self.generator(control) 
                            loss = criterion(xc_recon, control)
                            c_predict = self.dec(xc_recon)
                            c_loss = self.sce_loss(c_predict.to(self.device) ,c_target.to(self.device) )
                            # loss = loss+c_loss * 0.1
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                            loop.set_postfix(Loss=loss.item())
                        self.generator.eval()
                        val_total_loss = 0.0
                        loop = tqdm(enumerate( self.test_dataloader), total =len(self.test_dataloader))
                        with torch.no_grad():
                            for i,data in loop:
                                x = data['data'].to(self.device)
                                c_target = data['obs']
                                x_recon, z = self.generator(x)
                                loss = criterion(x_recon, x)
                                c_predict = self.dec(x_recon)
                                c_loss = self.sce_loss(c_predict.to(self.device),c_target.to(self.device))
                                # loss = loss+c_loss * 0.1
                                val_total_loss += loss.item() 
                        train_avg_loss = total_loss / len(self.train_dataloader)    
                        val_avg_loss = val_total_loss / len(self.test_dataloader)       
                        print(f"Epoch {epoch+1}/{100}")
                        print(f"  train loss: {train_avg_loss:.6f}")
                        print(f"  val loss: {val_avg_loss:.6f}")
                        if val_avg_loss < best_val_loss:
                            best_val_loss = val_avg_loss
                            early_stop_counter = 0 
                            checkpoint = {
                                'epoch': epoch + 1,
                                'model_state_dict': self.generator.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_val_loss': best_val_loss
                            }
                            torch.save(checkpoint, os.path.join('./_weight_modelVAE/', self.dataset_name +'_716_DAE_generator_best.pkl'))
                            print(f"save the beat model and valid loss: {best_val_loss:.6f}）")
                        else:
                            early_stop_counter += 1
                            print(f"  early stop counts: {early_stop_counter}/{20}")
                            if early_stop_counter >= 20:
                                print("  early stop and stop the trainning")
                                break

                        scheduler.step(val_avg_loss)
                    

        self.params = filter(lambda p: p.requires_grad, self.model.parameters())
        # paramsPGM = filter(lambda p: p.requires_grad,  self.model.parameters())
        GRADIENT_ACCUMULATION = 1
        AMP_DTYPE = "float16"
        self.scaler = GradScaler(enabled=(AMP_DTYPE == 'float16'))
        self.optimPGM = torch.optim.Adam(
            self.params, lr=lr, weight_decay= weight_decay) # consider changing the param. like weight_decay, eps, etc.
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(self.optimPGM, step_size=1, gamma=0.5)

# )
        # self.model = torch.compile(self.model)
        if model_path is not None:
             self._load_checkpoint(model_path,strict=False)

        # if da_mode ==True:
        self.model.module.freeze_pretrained_modules()    
        total_trainable = 0  
        total_frozen = 0    

        for name, module in self.model.named_modules():
            
            params = list(module.parameters(recurse=False))
            if len(params) > 0:
                
                trainable = sum(p.numel() for p in params if p.requires_grad)
                frozen = sum(p.numel() for p in params if not p.requires_grad)

                
                total_trainable += trainable
                total_frozen += frozen

        
        #
        print("=" * 50)
        print(f"gobal params:")
        print(f"trainable: {total_trainable}")
        print(f"freezed: {total_frozen}")
        print(f"total: {total_trainable + total_frozen}")

        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[AMP_DTYPE]
        self.da_save_dir = 'DA730COPD'
        weight_path = './_weight_modelVAE/'+self.da_save_dir+'/epoch_3_best.pth'

        if os.path.exists(weight_path):
            self.G.load_state_dict(torch.load(weight_path)['G_state_dict'])

        else:
            if os.path.exists('./_weight_modelVAE/'+self.da_save_dir):
                self.adapt(self.generator.to(self.device)) 
            else:
                os.mkdir('./_weight_modelVAE/'+self.da_save_dir)
                self.adapt(self.generator.to(self.device)) 

        self.G.eval()
        self.G = self.G.cpu()
        for self.epoch in range(self.n_epochs):
            self.model.train()
            if da_mode == True:
                loop = tqdm(enumerate(zip(cycle(self.target_train_dataloader), self.train_dataloader)), total =len(self.train_dataloader))
                for i, data in loop:
                    self.model.zero_grad()
                    (control, target) = data[1]['features']
                    encode_label = data[1]['label']
                    
                    # t_control = data[0]['data']
                    # (t_control, t_target) = data[0]['features']
                    # with torch.no_grad():
                    #     control = self.generator(control)
                    #     target = self.generator(target)
                    # with torch.no_grad():
                    #     control = self.G(control)
                    #     target = self.G(target)
                    min_size = min(t_control.shape[0],control.shape[0])
                    if min_size<10:
                        continue  
                    if (t_control.shape[0]!=control.shape[0]):

                        t_control = t_control[:min_size,:]
                        t_target = t_target[:min_size,:]
                        control = control[:min_size,:]
                        encode_label = encode_label[:min_size,:]
                        target = target[:min_size,:]

                    control = control.to(self.device, dtype=torch.float32)
                    
                    target = target.to(self.device, dtype=torch.float32)
                    
                    encode_label = encode_label.to(self.device, dtype=torch.float32)
                    b_size = control.size(0)
                    noise = self.make_noise(b_size, 20)
                    tb_size = t_control.size(0)
                    t_noise = self.make_noise(tb_size, 20)
                    gene_reconstructions,emb_gene =  self.model(control, encode_label, noise)
                    with torch.no_grad():
                        t_gene_reconstructions,t_emb_gene = self.model(t_control.to(self.device, dtype=torch.float32), encode_label, t_noise)

                    if i % GRADIENT_ACCUMULATION != 0:
                        with autocast(device_type='cuda', dtype=ptdtype):
                            dim = gene_reconstructions.size(1) // 2
                            gene_means = gene_reconstructions[:, :dim]
                            gene_vars = gene_reconstructions[:, dim:]
                            gene_vars = F.softplus(gene_vars)
                            if set(['GUSS']).issubset(self.loss):
                                reconstruction_loss = self.criterion(input=gene_means, target=target, var=gene_vars)
                                s_dist = normal.Normal(
                                    torch.clamp(
                                        torch.Tensor(gene_means),
                                        min=1e-3,
                                        max=1e3,
                                    ),
                                    torch.clamp(
                                        torch.Tensor(gene_vars.sqrt()),
                                        min=1e-3,
                                        max=1e3,
                                    )          
                                )

                            loss = (reconstruction_loss) / GRADIENT_ACCUMULATION
                        self.scaler.scale(loss).backward()
                    else:
                        with autocast(device_type='cuda', dtype=ptdtype):
                            dim = gene_reconstructions.size(1) // 2
                            gene_means = gene_reconstructions[:, :dim]
                            gene_vars = gene_reconstructions[:, dim:]
                            gene_vars = F.softplus(gene_vars)

                            if set(['GUSS']).issubset(self.loss):
                                reconstruction_loss = self.criterion(input=gene_means, target=target, var=gene_vars)
                                s_dist = normal.Normal(
                                    torch.clamp(
                                        torch.Tensor(gene_means),
                                        min=1e-3,
                                        max=1e3,
                                    ),
                                    torch.clamp(
                                        torch.Tensor(gene_vars.sqrt()),
                                        min=1e-3,
                                        max=1e3,
                                    )           
                                )
                                dist = normal.Normal(
                                    torch.clamp(
                                        torch.Tensor(t_gene_reconstructions[:, :dim]),
                                        min=1e-3,
                                        max=1e3,
                                    ),
                                    torch.clamp(
                                        torch.Tensor(F.softplus(t_gene_reconstructions[:, dim:]).sqrt()),
                                        min=1e-3,
                                        max=1e3,
                                    )           
                                )


                            loss = (reconstruction_loss) / GRADIENT_ACCUMULATION
                        # print(con_mmd_loss)
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimPGM)
                        # optimizer update
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(),clip_value=1.0)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), int(1e2))
                        self.scaler.step(self.optimPGM)
                        self.scaler.update()
                        self.optimPGM.zero_grad()
                    # Update PGM
                    # self.optimPGM.step()
                    nb_sample = dist.sample().cpu().numpy()
                    yp_m = nb_sample.mean(0)
                    yp_v = nb_sample.var(0)

                    y_true = t_target.cpu().numpy()
                    yt_m = y_true.mean(axis=0)
                    yt_v = y_true.var(axis=0)
                    
                    
                    r2_score_mean = r2_score(yt_m, yp_m)
                    #self.scheduler_autoencoder.step()
                    snb_sample = s_dist.sample().cpu().numpy()
                    yp_m = snb_sample.mean(0)
                    yt_m = target.cpu().numpy().mean(axis=0)
                    sr2_score_mean = r2_score(yt_m, yp_m)
                    # Output training stats               
                    # Save Losses for plotting later
                    self.PGM_losses.append(reconstruction_loss.item())


                    loop.set_description(f'Epoch [{self.epoch}/{self.n_epochs}] [{i}/{len(self.train_dataloader)}]')
                    #loop.set_postfix(Loss_NB=nb_loss.item(), Loss_MSE=mse_loss.item())
                    loop.set_postfix(Loss=reconstruction_loss.item(),sR2=sr2_score_mean,R2=r2_score_mean)
            else:

                
                loop = tqdm(enumerate(self.train_dataloader), total =len(self.train_dataloader))
                for i, data in loop:
                    self.model.zero_grad()
                    (control, target) = data['features']
                    # batch_id = data['batch'][0]
                    # t_data = data[0]['data']
                    encode_label = data['label']
                    gene_token = data['top_gene_tokens']
                    # with torch.no_grad():
                    #     control = self.G(control)
                    #     target = self.G(target)
                    x_indices = data['top_gene_indices']
                    x_raw = data['control_raw']

                    x_raw = x_raw.to(self.device, dtype=torch.float32)
                    control = control.to(self.device, dtype=torch.float32)
                    target = target.to(self.device, dtype=torch.float32)

                    
                    encode_label = encode_label.to(self.device, dtype=torch.float32)
                    b_size = control.size(0)

                    noise = self.make_noise(b_size, 20)



                    gene_reconstructions , latent  =  self.model(gene_token,x_indices,x_raw,control, encode_label, noise)
                    
                    cov_drug_list = data['cov_drug']
                    size = set(cov_drug_list)

                    s_loss = 0
                    for j in size:
                        s = cosine_similarity(latent[np.asarray(cov_drug_list) == j ,:].cpu().detach().numpy())
                        s = 1-s
                        s_loss += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                    if(self.device=="cuda"):
                        s_loss = torch.tensor(s_loss).cuda()
                    else:
                        s_loss = torch.tensor(s_loss).cpu() 
                    s_loss.requires_grad_(True)

                    if i % GRADIENT_ACCUMULATION != 0:
                        with autocast(device_type='cuda', dtype=ptdtype):
                            dim = gene_reconstructions.size(1) // 2
                            gene_means = gene_reconstructions[:, :dim]
                            gene_vars = gene_reconstructions[:, dim:]
                            # gene_vars = torch.exp(gene_vars)
                            gene_vars = F.softplus(gene_vars)
                            if set(['GUSS']).issubset(self.loss):
                                reconstruction_loss = self.criterion(input=gene_means, target=target, var=gene_vars)


                                dist = normal.Normal(
                                    torch.clamp(
                                        torch.Tensor(gene_means),
                                        min=1e-3,
                                        max=1e3,
                                    ),
                                    torch.clamp(
                                        torch.Tensor(gene_vars.sqrt()),
                                        min=1e-3,
                                        max=1e3,
                                    )           
                                )
                            if set(['NB']).issubset(self.loss):
                                reconstruction_loss =  self.criterion(torch.cat([gene_means,gene_vars],dim=1), target)

                                counts, logits = self._convert_mean_disp_to_counts_logits(
                                    torch.clamp(
                                        torch.Tensor(gene_means),
                                        min=1e-3,
                                        max=1e3,
                                    ),
                                    torch.clamp(
                                        torch.Tensor(gene_vars),
                                        min=1e-3,
                                        max=1e3,
                                    )
                                )

                                dist = NegativeBinomial(
                                    total_count=counts,
                                    logits=logits
                                )

                            nb_sample = dist.sample()      

                            loss = (reconstruction_loss+s_loss) / GRADIENT_ACCUMULATION

                        self.scaler.scale(loss).backward()

                    else:
                        with autocast(device_type='cuda', dtype=ptdtype):
                            dim = (gene_reconstructions.size(1)) //3
                            gene_means = gene_reconstructions[:, :dim]  
                            gene_vars = gene_reconstructions[:, dim:2*dim]  
                            gene_zi_logits = gene_reconstructions[:, 2*dim:]  
                            gene_vars = F.softplus(gene_vars)

                            if set(['GUSS']).issubset(self.loss):

                                reconstruction_loss = self.criterion(
                                input_means=gene_means,
                                input_vars=gene_vars,
                                input_zi_logits=gene_zi_logits,
                                target=target
                            )
                                zi_prob = torch.sigmoid(gene_zi_logits)  # [batch, genes]：zero prob
                                log_normal_dist = Normal(
                                    loc=torch.clamp(gene_means, min=1e-3, max=1e3),
                                    scale=torch.clamp(torch.sqrt(gene_vars), min=1e-3, max=1e3)
                                )

                            if set(['ZINB']).issubset(self.loss):
                                mu, theta, zi_logits = torch.split(gene_reconstructions, 17911, dim=1)
                                mu = F.softplus(mu) + 1e-6 
                                theta = F.softplus(theta) + 1e-6 
                                
                                dist = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=zi_logits)
                                reconstruction_loss = -dist.log_prob(target).mean()  

                            def euclidean_dist_matrix(x):               # x: [N, d]
                                n = x.size(0)
                                xx = torch.mm(x, x.t())                 # [N, N]
                                rx = xx.diag().unsqueeze(0)             # [1, N]
                                dist = rx.t() + rx - 2 * xx             # (xi-xj)^2 = |xi|^2 + |xj|^2 - 2 xi·xj
                                dist = torch.sqrt(torch.clamp(dist, min=1e-8))
                                return dist                             # [N, N]

                            target_np = target.cpu().detach().numpy()
                            edgeList   = calculateKNNgraphDistanceMatrix(target_np, distanceType='euclidean', k=30)
                            listResult, size = generateLouvainCluster(edgeList)

                            _ = latent                                 # 保留梯度
                            for i in range(size):
                                mask = torch.tensor(listResult == i, device=control.device)
                                n = mask.sum().item()
                                if n < 2:
                                    continue
                                group = _[mask]                        # [n, d]
                                dist  = euclidean_dist_matrix(group)  # [n, n]
                                triu_mask = torch.triu(torch.ones_like(dist), diagonal=1).bool()
                                avg_dist = dist[triu_mask].mean()
                                s_loss += -avg_dist                  

                            nb_sample = sample_ziln(
                                zi_prob=zi_prob,
                                log_normal_dist=log_normal_dist,
                                device=gene_means.device,  
                                truncate_max=1e4 
                            )

                            loss = (reconstruction_loss+s_loss) / GRADIENT_ACCUMULATION
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimPGM)

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), int(1e2))
                        self.scaler.step(self.optimPGM)
                        self.scaler.update()
                        self.optimPGM.zero_grad()
                # Update PGM
                # self.optimPGM.step()
                    nb_sample = nb_sample.cpu().numpy()
                    yp_m = nb_sample.mean(0)
                    yp_v = nb_sample.var(0)

                    y_true = target.cpu().numpy()
                    yt_m = y_true.mean(axis=0)
                    yt_v = y_true.var(axis=0)


                    r2_score_mean = r2_score(yt_m, yp_m)

                    # Output training stats               
                    # Save Losses for plotting later
                    self.PGM_losses.append(reconstruction_loss.item())


                    loop.set_description(f'Epoch [{self.epoch}/{self.n_epochs}] [{i}/{len(self.train_dataloader)}]')
                    #loop.set_postfix(Loss_NB=nb_loss.item(), Loss_MSE=mse_loss.item())
                    loop.set_postfix(Loss=reconstruction_loss.item(),R2=r2_score_mean)
            if da_mode == True:
                # self.model.eval()
                loop_v = tqdm(enumerate(zip(cycle(self.target_test_dataloader), self.test_dataloader)), total =len(self.test_dataloader))
                self.r2_sum_mean = 0
                self.r2_sum_var = 0
                self.score_sum = 0
                self.r2_sum_mean_de = 0
                self.r2_sum_var_de = 0
                self.mse_sum = 0
                for j, vdata in loop_v:
                    (control, target) = vdata[0]['features']
                    encode_label = vdata[0]['label']

                    control = control.to(self.device, dtype=torch.float32)
                    
                    target = target.to(self.device, dtype=torch.float32)

                    encode_label = encode_label.to(self.device, dtype=torch.float32)
                    b_size = control.size(0)

                    noise = self.make_noise(b_size, 20)
                    with torch.no_grad():
                        gene_reconstructions,emb_gene= self.model(control, encode_label, noise)
                    dim = gene_reconstructions.size(1) // 2
                    gene_means = gene_reconstructions[:, :dim]
                    gene_vars = gene_reconstructions[:, dim:]
                    gene_vars = F.softplus(gene_vars)
                    if set(['GUSS']).issubset(self.loss):
                        reconstruction_loss = self.criterion(input=gene_means, target=target, var=gene_vars)

                        dist = normal.Normal(
                            torch.clamp(
                                torch.Tensor(gene_means),
                                min=1e-3,
                                max=1e3,
                            ),
                            torch.clamp(
                                torch.Tensor(gene_vars.sqrt()),
                                min=1e-3,
                                max=1e3,
                            )           
                        )
                    nb_sample = dist.sample().cpu().numpy()
                    yp_m = nb_sample.mean(0)
                    yp_v = nb_sample.var(0)

                    y_true = target.cpu().numpy()
                    yt_m = y_true.mean(axis=0)
                    yt_v = y_true.var(axis=0)


                    r2_score_mean = r2_score(yt_m, yp_m)
                    self.r2_sum_mean += r2_score_mean
                    r2_score_var = r2_score(yt_v, yp_v)
                    self.r2_sum_var += r2_score_var       
                    eps = 1e-8
#
                    mse_score =  mean_squared_error(y_true, nb_sample)
                    self.mse_sum += mse_score
                    self.score_sum += (mse_score)

                    loop_v.set_description(f'Epoch [{self.epoch}/{self.n_epochs}] [{j}/{len(self.test_dataloader)}]')
                    loop_v.set_postfix(r2_score_mean=r2_score_mean, r2_score_var=r2_score_var, mse_score=mse_score)
                    # loop_v.set_postfix(r2_score_mean=r2_score_mean, r2_score_var=r2_score_var, mse_score=mse_score)
                self.r2_score_mean.append(self.r2_sum_mean/len(self.test_dataloader))
                self.r2_score_var.append(self.r2_sum_var/len(self.test_dataloader))
                self.score.append(self.score_sum/len(self.test_dataloader))
                print('mean mse of validation datastes:', self.mse_sum/len(self.test_dataloader))
                print('mean score of validation datastes:', self.score[-1])
                print('mean r2 of validation datastes:', self.r2_score_mean[-1])

                self.scheduler_autoencoder.step()                       

            else:
                self.model.eval()
                loop_v = tqdm(enumerate(self.test_dataloader), total =len(self.test_dataloader))
                self.r2_sum_mean = 0
                self.r2_sum_var = 0
                self.score_sum = 0
                self.r2_sum_mean_de = 0
                self.r2_sum_var_de = 0
                self.mse_sum = 0
                for j, vdata in loop_v:
                    
                    (control, target) = vdata['features']

                    gene_token = vdata['top_gene_tokens']
                    x_indices = vdata['top_gene_indices']
                    x_raw = vdata['control_raw'].to(self.device, dtype=torch.float32)

                    encode_label = vdata['label']
                    # data_cov_drug = vdata['cov_drug']
                    control = control.to(self.device, dtype=torch.float32)
                    if set(['NB']).issubset(self.loss):
                        control = torch.log1p(control)
                    target = target.to(self.device, dtype=torch.float32)

                    encode_label = encode_label.to(self.device, dtype=torch.float32)
                    b_size = control.size(0)
                    # batch_id = vdata['batch'][0]
                    noise = self.make_noise(b_size, 20)
                    with torch.no_grad():
                        gene_reconstructions,latent =  self.model(gene_token,x_indices,x_raw,control, encode_label, noise)
                        dim = (gene_reconstructions.size(1)) //3
                        gene_means = gene_reconstructions[:, :dim]  
                        gene_vars = gene_reconstructions[:, dim:2*dim]  
                        gene_zi_logits = gene_reconstructions[:, 2*dim:]  
                        
                        gene_vars = F.softplus(gene_vars)


                    if set(['GUSS']).issubset(self.loss):

                        reconstruction_loss = self.criterion(
                                input_means=gene_means,
                                input_vars=gene_vars,
                                input_zi_logits=gene_zi_logits,
                                target=target
                            )
                        zi_prob = torch.sigmoid(gene_zi_logits)  # [batch, genes]：零概率
                        log_normal_dist = Normal(
                                    loc=torch.clamp(gene_means, min=1e-3, max=1e3),
                                    scale=torch.clamp(torch.sqrt(gene_vars), min=1e-3, max=1e3)
                                )
                    if set(['NB']).issubset(self.loss):
                        reconstruction_loss = self.criterion(torch.cat([gene_means,gene_vars],dim=1), target)

                        counts, logits = self._convert_mean_disp_to_counts_logits(
                            torch.clamp(
                                torch.Tensor(gene_means),
                                min=1e-3,
                                max=1e3,
                            ),
                            torch.clamp(
                                torch.Tensor(gene_vars),
                                min=1e-3,
                                max=1e3,
                            )
                        )

                        dist = NegativeBinomial(
                            total_count=counts,
                            logits=logits
                        )

                    if set(['ZINB']).issubset(self.loss):
                        mu, theta, zi_logits = torch.split(gene_reconstructions, 17911, dim=1)
                        mu = F.softplus(mu) + 1e-6 
                        theta = F.softplus(theta) + 1e-6 

                        dist = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=zi_logits)
                    nb_sample = sample_ziln(
                            zi_prob=zi_prob,
                            log_normal_dist=log_normal_dist,
                            device=gene_means.device,  
                            truncate_max=1e4 
                        ).cpu().numpy()
                    yp_m = nb_sample.mean(0)
                    yp_v = nb_sample.var(0)

                    y_true = target.cpu().numpy()
                    yt_m = y_true.mean(axis=0)
                    yt_v = y_true.var(axis=0)


                    r2_score_mean = r2_score(yt_m, yp_m)
                    self.r2_sum_mean += r2_score_mean
                    r2_score_var = r2_score(yt_v, yp_v)
                    self.r2_sum_var += r2_score_var               

                    mse_score =  mean_squared_error(y_true, nb_sample)
                    self.mse_sum += mse_score
                    self.score_sum += (mse_score)

                    loop_v.set_description(f'Epoch [{self.epoch}/{self.n_epochs}] [{j}/{len(self.test_dataloader)}]')
                    loop_v.set_postfix(r2_score_mean=r2_score_mean, r2_score_var=r2_score_var, mse_score=mse_score)
                self.r2_score_mean.append(self.r2_sum_mean/len(self.test_dataloader))
                self.r2_score_var.append(self.r2_sum_var/len(self.test_dataloader))
                self.score.append(self.score_sum/len(self.test_dataloader))
                print('mean mse of validation datastes:', self.mse_sum/len(self.test_dataloader))
                print('mean score of validation datastes:', self.score[-1])
                print('mean r2 of validation datastes:', self.r2_score_mean[-1])

                self.scheduler_autoencoder.step()    
            if self.score[-1] < self.best_score:     
                self.patient = 0
                print("Saving best state of network...")
                print("Best State was in Epoch", self.epoch)
                self.best_state_dictG = self.model.module.state_dict()
                self._save_checkpoint(self.model_save_dir+self.split_key+'_scMCP_DrugGCL_go_da_drug_for_tahoe_1016_tagoebase_genecompass_DrugGCL_768_GUSS_load_best_epoch_all.pt', self.epoch, self.score[-1])
                # torch.save(self.best_state_dictG, self.model_save_dir+self.split_key+'_scMCP_coexpress_drug_tagoe1M_DrugGCL_768_load_best_epoch_all.pt')
                self.best_score = self.score[-1]
            elif self.patient <= 20:
                self.patient += 1   
            else:
                print("The mse of validation datastes has not improve in 20 epochs!")
                break

        
        loss_dict = {'Loss_PGM': self.PGM_losses}
        metrics_dict = {'r2':self.r2_score_mean, 'mse':self.score}
        loss_df = pd.DataFrame(loss_dict)
        metrics_df = pd.DataFrame(metrics_dict)
        loss_df.to_csv(self.model_save_dir+self.split_key+'loss_comb.csv')
        metrics_df.to_csv(self.model_save_dir+self.split_key+'metrics_comb.csv')
    def fisher_matrix_diag(model, loader):
        model.eval()
        fisher = {
       }
             
        for param_name, _ in model.named_parameters():
            fisher[param_name] = torch.zeros_like(model.state_dict()[param_name])

        for i, data in loader:
            (control, target) = data['features']
            control = control.to(self.device, dtype=torch.float32)
            encode_label = encode_label.to(self.device, dtype=torch.float32)
            b_size = control.size(0)

            noise = self.make_noise(b_size, 20)
            gene_reconstructions,_= model(control,encode_label,noise)
            dim = gene_reconstructions.size(1) // 2
            gene_means = gene_reconstructions[:, :dim]  
            gene_vars = F.softplus(gene_reconstructions[:, dim:]) 

            log_likelihood = -0.5 * torch.log(2 * torch.pi * gene_vars)  
            log_likelihood -= 0.5 * (target - gene_means) ** 2 / gene_vars
            total_log_likelihood = log_likelihood.sum()  

            model.zero_grad()
            total_log_likelihood.backward(retain_graph=True)

            for param_name, param in model.named_parameters():
                if param.grad is not None:  
                    fisher[param_name] += param.grad ** 2  

        num_samples = len(loader)
        for param_name in fisher:
            fisher[param_name] /= num_samples

        return fisher

    def ewc_train(model, loader, optimizer, criterion, fisher, prev_task_params, lamda=1000):
        model.train()
        for i, data in loader:
            optimizer.zero_grad()
            (control, target) = data['features']
            control = control.to(self.device, dtype=torch.float32)
            encode_label = encode_label.to(self.device, dtype=torch.float32)
            b_size = control.size(0)

            noise = self.make_noise(b_size, 20)
            gene_reconstructions,_= model(control,encode_label,noise)
            dim = gene_reconstructions.size(1) // 2
            gene_means = gene_reconstructions[:, :dim]
            gene_vars = gene_reconstructions[:, dim:]
            gene_vars = F.softplus(gene_vars)
            loss = self.criterion(input=gene_means, target=target, var=gene_vars)
            ewc_loss = 0
            for name, param in model.named_parameters():
                _loss = fisher[name] * (prev_task_params[name] - param).pow(2)
                ewc_loss += _loss.sum()
            loss += lamda * ewc_loss
            loss.backward()
            optimizer.step()
    @staticmethod
    def _anndataToTensor(adata: AnnData) -> torch.Tensor:
        data_ndarray = adata.X.A
        data_tensor = torch.from_numpy(data_ndarray)
        return data_tensor
    
    def make_noise(self, batch_size, shape, volatile=False):
        tensor = torch.randn(batch_size, shape)
        noise = Variable(tensor, volatile)
        noise = noise.to(self.device, dtype=torch.float32)
        return noise
    def adapt(self, generator: GeneratorAD):
        
        tqdm.write('Begin to correct data domain shifts...')
        self.D.train()
        self.G.train()
        self._train_DA_model(self.n_epochs,generator)

    @torch.no_grad()
    def _batch_map(self, ref: torch.Tensor, tgt: torch.Tensor, generator: GeneratorAD):

        generator.eval()
        ref_e = generator(ref)[0]
        tgt_e = generator(tgt)[0]

        dot_product_matrix = torch.mm(tgt_e, ref_e.t())
        max_indices = torch.argmax(dot_product_matrix, dim=1)
        mapped_tgt_e = tgt_e[max_indices]
        return ref, mapped_tgt_e.detach()
    @torch.no_grad()
    def _map(self, generator: GeneratorAD):
        ref_data = torch.Tensor(ref.X).to(self.device)
        tgt_data = torch.Tensor(tgt.X).to(self.device)

        generator.eval()
        ref_e = generator(ref_data)  
        tgt_e = generator(tgt_data)  

        batch_size_tgt = 128 
        batch_size_ref = 1024 
        max_indices = []

        for i in range(0, tgt_e.size(0), batch_size_tgt):
            tgt_batch = tgt_e[i:i+batch_size_tgt] 
            batch_max_indices = []
            for j in range(0, ref_e.size(0), batch_size_ref):
                ref_batch = ref_e[j:j+batch_size_ref] 
                dot_product = torch.mm(tgt_batch, ref_batch.t())
                max_vals, max_idx_in_batch = torch.max(dot_product, dim=1)
                max_idx_in_batch += j 
                batch_max_indices.append((max_vals, max_idx_in_batch))
            all_vals, all_indices = zip(*batch_max_indices)
            all_vals = torch.stack(all_vals, dim=1)  
            all_indices = torch.stack(all_indices, dim=1)
            global_max_idx = torch.argmax(all_vals, dim=1)
            final_indices = all_indices[torch.arange(B_tgt), global_max_idx]
            max_indices.append(final_indices.cpu())

        max_indices = torch.cat(max_indices, dim=0)
        mapped_ref_e = ref_e[max_indices].detach().cpu()
        return mapped_ref_e, tgt_data.cpu()

    def weight_init(self, m):  
        # initialize the weights of the model
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                m.weight.data.normal_(1, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
    

    @staticmethod
    def pearson_mean(data1, data2):
        sum_pearson_1 = 0
        sum_pearson_2 = 0
        for i in range(data1.shape[0]):
            pearsonr_ = pearsonr(data1[i], data2[i])
            sum_pearson_1 += pearsonr_[0]
            sum_pearson_2 += pearsonr_[1]
        return sum_pearson_1/data1.shape[0], sum_pearson_2/data1.shape[0]
    
    @staticmethod
    def r2_mean(data1, data2):
        sum_r2_1 = 0
        for i in range(data1.shape[0]):
            r2_score_ = r2_score(data1[i], data2[i])
            sum_r2_1 += r2_score_           
        return sum_r2_1/data1.shape[0]
    def _save_checkpoint(self, path, epoch, loss):
        """Save the whole model state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.best_state_dictG,
            'optimizer_state_dict': self.optimPGM.state_dict(),
            'scheduler_state_dict': self.scheduler_autoencoder.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_score': loss
        }
        torch.save(checkpoint, path)
        print(f"Checkpoints have been save to {path} (Epoch: {epoch}, Loss: {loss:.4f})")
    def _load_checkpoint(self, path,strict=True):
        """Load the whole model state"""
        checkpoint = torch.load(path)
        if strict:
            self.model.module.load_state_dict(checkpoint['model_state_dict'],strict=False)
            # self.optimPGM.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler_autoencoder.load_state_dict(checkpoint['scheduler_state_dict'])
            # self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            # self.current_epoch = checkpoint['epoch'] + 1
            # self.best_score = checkpoint['best_score']
            # print(f"{path} have been loaded (Epoch: {self.current_epoch-1})")
        else:
            self.model.module.load_state_dict(checkpoint['model_state_dict'],strict=False)
