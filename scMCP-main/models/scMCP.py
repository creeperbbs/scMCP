# -*- coding: utf-8 -*-
# @Author: Hongzhi Yao
# @Date:   2025-04-21 09:05:55
# @Last Modified by:   Hongzhi Yao
# @Last Modified time: 2025-10-16 16:33:14
from torch_geometric.nn import SGConv
# from .drug_gene_rotary_attention import DrugGeneAttention
from .Performer import *
import torch.nn as nn
from torch_geometric.nn import SGConv
import pandas as pd
import os 

from torch import Tensor
from tqdm import tqdm
from igraph import *
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from .graph_networks import GeneGINet
import sys
sys.path.append("/GeneCompass-main/")
from collections import OrderedDict
from genecompass import BertForMaskedLM, BertForSequenceClassification,BertForMaskedLMWithSEMA
from genecompass.utils import load_prior_embedding  


class TwoLayerMLP(nn.Module):
    def __init__(self):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(1, 50)   # Input size: 1, Output size: 100
        self.fc2 = nn.Linear(50, 128) # Input size: 100, Output size: 128
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

knowledges = dict()


out = load_prior_embedding(
# name2promoter_human_path, name2promoter_mouse_path, id2name_human_mouse_path,
# token_dictionary
# prior_embedding_path
)
knowledges['promoter'] = out[0]
knowledges['co_exp'] = out[1]
knowledges['gene_family'] = out[2]
knowledges['peca_grn'] = out[3]
knowledges['homologous_gene_human2mouse'] = out[4]
class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class FeedForward(nn.Module):
    def __init__(self, in_dim,out_dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(in_dim, in_dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(in_dim * mult, out_dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nheads, hidden_dim=1024, dropout=0.3) -> None:
        super().__init__()

        self.attention = SelfAttention(d_model)

        self.norm = nn.ModuleList([
            copy.deepcopy(nn.LayerNorm(d_model)) for _ in range(2)
        ])

        self.fc = Chunk(8, FeedForward(d_model,d_model, mult = 1, dropout = dropout, glu = True), along_dim = 1)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v):
        attn = self.attention(q, context=k)

        x = self.dropout(self.norm[0](attn + q))
        f = self.fc(x)
        x = self.dropout(self.norm[1](x + f))
        return x
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)    
import copy
class gene_gene_operator(torch.nn.Module):
    def __init__(self, input_genes,
                 grn_emb_list =None, 
                 ppi_emb_list  =None,  

                 device = 'cpu',
        top_seq_len = 5000,
        base_dim = 128,                                # dim of tokens

        emb_dropout = 0.,

        coexpress_network = None,
                 gene_emb_adapt =False,
                 output_dim = 200):
        super(gene_gene_operator, self).__init__()
        # Create a mapping from new_indices to original_indices
        self.grn_emb_list = grn_emb_list
        self.ppi_emb_list = ppi_emb_list
        self.device = device
        self.gene_emb_adapt = gene_emb_adapt
        self.coexpress_network = coexpress_network
        if coexpress_network is not None:
            num_genes = 17911
            num_layer =2
            emb_dim = 128
            feat_dim = 128
            self.geneGINet = GeneGINet(num_genes=num_genes, num_layer=num_layer, emb_dim=emb_dim, feat_dim=feat_dim)
            self.edge_index = coexpress_network.edge_index
            self.edge_weight = coexpress_network.edge_weight
        self.fc1 = nn.Linear(256, base_dim)
        self.fc2 = nn.Linear(128, base_dim)
        self.base_dim = base_dim
        self.top_seq_len = top_seq_len
        self.dropout1 = nn.Dropout(emb_dropout)
        # self.dropout2 = nn.Dropout(emb_dropout)
        self.input_gene_num = input_genes
        # self.token_emb = nn.Embedding(17911, 128)
        self.token_emb =TwoLayerMLP()
        # self.fc1 = nn.Linear(384, 128) # Input size: 100, Output size: 200 Chunk(4, FeedForward(384,128, mult = 4, dropout = emb_dropout, glu = True), along_dim = 1)
        adapt_start_layer = 0
        adapt_end_layer = 3
        buffer_size = 128
        if self.gene_emb_adapt:
                # self.freeze_singlecell_model()
                self.adapt_start_layer = adapt_start_layer
                self.adapt_end_layer = adapt_end_layer
                self.buffer_size = buffer_size
                self.sema_modules = nn.ModuleList()
                hidden_list = [128,128,128]
                for layer_id, h in enumerate(hidden_list):  
                    sema = SEMAModules(
                        adapt_start_layer=adapt_start_layer,  
                        adapt_end_layer=adapt_end_layer,     
                        d_model=h,  
                        layer_id=layer_id, 
                        writer=None  
                    ).to(self.device)  #
                    self.sema_modules.append(sema)  # 
        
        self.attr_gene = False
        # self.LayerNorm1 = nn.LayerNorm(128, eps=1e-3)
        self.LayerNorm2 = nn.LayerNorm(128)

        dim = 1

        dr_rate = 0.3
        if self.attr_gene==True:
            self.top_gene_gene_attn = nn.ModuleList([
            TransformerLayer(128, 4, 64, dr_rate) 
                for _ in range(2)
            ])
            self.low_gene_gene_attn = nn.ModuleList([
            TransformerLayer(128, 4, 32, dr_rate) 
                for _ in range(1)
            ])
        self.output_dim = output_dim
        # if self.output_dim != 200:
        #     try:
        #         self.feedforward = nn.Linear(self.scfm.base_dim, self.output_dim)
        #     except AttributeError:
        #         self.feedforward = nn.Linear(scfm_hyper_params['dim'], self.output_dim)
        self.concat_embeddings = nn.Sequential(OrderedDict([
                    ("cat_fc", nn.Linear(768+128+128, 768)),
                    ("cat_ln", nn.LayerNorm(768)),
                    ("cat_gelu", QuickGELU()),
                    ("cat_proj", nn.Linear(768,768))
                ]))        

    def forward(self, x,x_raw,x_indices,**kwargs):
        output = {"merged_encodings": None}
        b, n = x.shape[0],x.shape[1]
        # _, indices = torch.sort(x, dim=1, descending=True)
        # d, L = self.base_dim, self.top_seq_len
        # _, top_indices = torch.topk(x, L, dim=1)
        # raw_x=x.unsqueeze(2)
        x_raw = self.token_emb(x_raw.unsqueeze(2))
        
        if self.attr_gene != False:
            gene_emb = self.grn_emb_list + self.ppi_emb_list
            gene_emb = torch.tensor(gene_emb, dtype=torch.float32)
            gene_emb = gene_emb.unsqueeze(0)
            gene_emb = gene_emb.repeat(x.shape[0], 1, 1).to(self.device)
            x_top = torch.gather(x, 1, top_indices.unsqueeze(-1).expand(-1, -1, d))
            gene_emb_top = torch.gather(gene_emb, 1, top_indices.unsqueeze(-1).expand(-1, -1, d))
            top_mask = torch.zeros(b, n, dtype=torch.bool, device=x.device)
            top_mask.scatter_(1, top_indices, True)

            x_low = x.masked_select(~top_mask.unsqueeze(-1)).view(b, n - L, d)
            gene_emb_low = gene_emb.masked_select(~top_mask.unsqueeze(-1)).view(b, n - L, d)
            for layer in self.top_gene_gene_attn:
                x_top = layer( x_top.float(),gene_emb_top.float(),gene_emb_top.float())
            for layer in self.low_gene_gene_attn:
                x_low = layer( x_low.float(),gene_emb_low.float(),gene_emb_low.float())
            x_merged = torch.zeros(b, n, d, device=x.device)

            x_merged.scatter_(1, top_indices.unsqueeze(-1).expand(-1, -1, d), x_top)

            low_mask = ~top_mask
            x_merged.masked_scatter_(low_mask.unsqueeze(-1), x_low.reshape(-1, d))
            x = x + x_merged
        else:
            gene_emb = torch.cat([
                torch.tensor(self.grn_emb_list, dtype=torch.float32, device=self.device),
                torch.tensor(self.ppi_emb_list, dtype=torch.float32, device=self.device)
            ], dim=1)
            B = x.shape[0]  
            fc1_out = self.fc1(gene_emb.expand(B, -1, -1) )

            selected_emb = fc1_out[torch.arange(B)[:, None], x_indices]
            x = torch.cat([x ,selected_emb],dim=2)
            del selected_emb  
            if self.coexpress_network is not None:
                try:
                    current_device = x.device

                    if self.edge_index.numel() > 0:  
                        max_idx = torch.max(self.edge_index)
                        min_idx = torch.min(self.edge_index)
                        node_count = x_raw.size(1)  

                        if max_idx >= node_count:
                            raise ValueError(f"edge_index: max_idx={max_idx} >=nodes={node_count}")
                        if min_idx < 0:
                            raise ValueError(f"edge_index: min_idx={min_idx}")

                    geneGI_emb = self.geneGINet.to(current_device)(
                        x_raw.to(current_device), 
                        self.edge_index.to(current_device), 
                        self.edge_weight.to(current_device)
                    )

                except RuntimeError as e:
                    print(f"\n===== geneGINet mediate RuntimeError =====")
                    print(f"error: {str(e)}")
                    print(f"x_raw detailed information: shape={x_raw.shape}, device={x_raw.device}, datatype={x_raw.dtype}")
                    print(f"edge_index detailed information: shape={self.edge_index.shape}, device={self.edge_index.device}, data type={self.edge_index.dtype}")
                    print(f"edge_weight detailed information: shape={self.edge_weight.shape}, device={self.edge_weight.device}, data type={self.edge_weight.dtype}")
                    if self.edge_index.numel() > 0:
                        print(f"edge_index index range: [{torch.min(self.edge_index)}, {torch.max(self.edge_index)}], node numbers={x_raw.size(1)}")
                    print("=========================================\n")
                    raise
                except ValueError as e:
                    print(f"\n===== verify failure =====")
                    print(f"error: {str(e)}")
                    print("==========================\n")
                    raise
                selected_geneGI_emb = geneGI_emb[torch.arange(B)[:, None], x_indices]
            x = torch.cat([x ,selected_geneGI_emb],dim=2)
        x_emb=self.concat_embeddings(x)
        # x_emb = self.LayerNorm2(x)
        # x_emb = self.dropout1(x)  
        output["merged_encodings"] = x_emb
        return output

class MLP(nn.Module):
    def __init__(self, hidden_dim=None, batch_norm=True,  last_layer_act='linear'):
        super(MLP, self).__init__()
        self.last_layer_act = last_layer_act
        layers = []
        for s in range(len(hidden_dim) - 1):
            layers = layers + [
                torch.nn.Linear(hidden_dim[s], hidden_dim[s + 1]),
                torch.nn.BatchNorm1d(hidden_dim[s + 1])
                if batch_norm and s < len(hidden_dim) - 1 else None,
                torch.nn.ReLU()
            ]
        layers = [l for l in layers if l is not None][:-1]
        self.EnE = torch.nn.Sequential(*layers)
        self.relu=nn.ReLU()
    def forward(self, x):
        output = self.EnE(x)
        if self.last_layer_act=='ReLU':
            output = self.relu(output)
        return output



class scMCP(torch.nn.Module):
    def __init__(self, 
                 num_genes=2048,
                 uncertainty=True,
                 num_gnn_layers=None,
                 decoder_hidden_size=None,
                 num_gene_gnn_layers=None,
                 gene_mask=None,
                 input_genes_ens_ids=None,
                 scfm_genes_ens_ids=None,
                 drug_dim=None,
                 drug_gene_attr =False,
                 n_size = 20,
                 adaptor = None,
                 coexpress_network=None,
                 gene_emb_dim=768,
                 dropout_rate=0.3,
                 hidden_size = 200,
                 d_model = 128,
                 da_mode=False,
                 gene_emb_adapt = False,
                adapt_start_layer=0,  
                 adapt_end_layer=1,   
                 buffer_size=128,     
                 pos_emb_graph = 'co_expression',
                 grn_node2vec_file='/emb_grn/grn_emb_total.pkl',
                 ppi_node2vec_file='/emb_ppi/ppi_emb_total.pkl',
                 model_type = 'ppi_grn_mode'
                ):
        super(DGFM, self).__init__()
        # self.args = args 
#G=11350++978,N=256
        self.device='cuda'
        if gene_mask is not None:
            self.gene_mask = gene_mask
        self.num_genes = num_genes
        self.hidden_size = hidden_size
        self.uncertainty = uncertainty
        self.num_layers = num_gnn_layers
        self.indv_out_hidden_size = decoder_hidden_size
        self.num_layers_gene_pos = num_gene_gnn_layers
        self.pert_emb_lambda = 0.2
        self.input_genes_list = input_genes_ens_ids
        self.scfm_genes_list = scfm_genes_ens_ids
        

        self.coexpress_network = coexpress_network
        self.gene_emb_adapt = gene_emb_adapt

        try:
            with open(grn_node2vec_file, 'rb') as handle:
                self.grn_node2vec_embedding_dict = pickle.load(handle)
        except ValueError:
            import pickle5
            with open(grn_node2vec_file, 'rb') as handle:
                self.grn_node2vec_embedding = pickle5.load(handle)
        try:
            with open(ppi_node2vec_file, 'rb') as handle:
                self.ppi_node2vec_embedding_dict = pickle.load(handle)
        except ValueError:
            import pickle5
            with open(ppi_node2vec_file, 'rb') as handle:
                self.ppi_node2vec_embedding_dict = pickle5.load(handle)

        self.top_seq_num = 5000   

        self.emb_trans = nn.ReLU()
        
        # self.transform = nn.ReLU()
        dr_rate = 0.3

        self.drug_gene_attr = drug_gene_attr
        if self.drug_gene_attr != False:
            self.top_drug_gene_attn = DrugGeneAttention(128, 128, n_layers=2, n_heads=4, pf_dim=256,
                                                    dropout=dropout_rate, device="cuda")
            self.low_drug_gene_attn = DrugGeneAttention(128, 128, n_layers=1, n_heads=4, pf_dim=64,
                                                    dropout=dropout_rate, device="cuda")
        # decoder shared MLP
        self.recovery_w = MLP([gene_emb_dim, hidden_size*2, hidden_size], last_layer_act='linear')


        self.relu = nn.ReLU()

        checkpoint_path = '/home/MBDAI206AA201/jupyter/yhz/sc/scdata/GeneCompass-main/pretrained_models/GeneCompass_Base/'
        self.bert = BertForMaskedLMWithSEMA.from_pretrained(
            checkpoint_path,
            knowledges=knowledges,
            ignore_mismatched_sizes=True,
        ).to("cuda")
        freeze_layers = 12
        for param in self.bert.parameters():
            param.requires_grad = False

        # for param in self.bert.bert.encoder.layer[11].parameters():
        #     param.requires_grad = True  
        self.downdecoder =  nn.Linear(768+128, 1, bias=True)
        # self.downdecoder = nn.Sequential(OrderedDict([
        #             ("cat_fc", nn.Linear(768+64, 768)),
        #             ("cat_ln", nn.LayerNorm(768)),
        #             ("cat_gelu", QuickGELU()),
        #             ("cat_proj", nn.Linear(768,768))
        #         ]))   
        self.pretrain_vae = False
        self.da_mode = da_mode
        self.bn_emb = nn.BatchNorm1d(gene_emb_dim)
        # self.bn_pert_base = nn.BatchNorm1d(num_genes)
        self.pert_base_trans = nn.ReLU()
        self.bn_pert_base_trans = nn.BatchNorm1d(d_model)
        self.comb_encoder =  nn.Sequential()
        self.comb_encoder.add_module(name="Lc0", module=nn.Linear(drug_dim, d_model*2, bias=False))
        self.comb_encoder.add_module(name="Lc1", module=nn.Linear(d_model*2, d_model))
        self.encoder = nn.Sequential()
        self.encoder.add_module(name="Le0", module=nn.Linear(self.num_genes, d_model*2, bias=False))
        self.encoder.add_module(name="Le1", module=nn.Linear(d_model*2, d_model*1))
        # self.encoder.add_module("Ne1", module=nn.BatchNorm1d(d_model*2))
        # self.encoder.add_module(name="Ae1", module=nn.LeakyReLU(negative_slope=0.3))
        # self.encoder.add_module(name="De1", module=nn.Dropout(p=dropout_rate))
        # self.encoder.add_module(name="Le2", module=nn.Linear(d_model*2, d_model))
        # self.decoder = nn.Linear(hidden_size+drug_dim+10, self.num_genes*2)
        self.decoder = nn.Sequential()
        # self.decoder.add_module("Nd0", module=nn.BatchNorm1d(d_model+d_model+n_size))
        self.decoder.add_module(name="Ld0", module=nn.Linear(d_model+d_model*1+n_size, d_model, bias=False))
        self.decoder.add_module("Nd0", module=nn.BatchNorm1d(d_model))
        self.decoder.add_module(name="Ad0", module=nn.LeakyReLU(negative_slope=0.3))
        self.decoder.add_module(name="Dd0", module=nn.Dropout(p=dropout_rate))
        self.decoder.add_module(name="Ld1", module=nn.Linear(d_model, 17911*3)) 
        

        self.adapt_same = False
        if self.adapt_same:
            self.intermediate = BertIntermediate(d_model,d_model)
            self.bertOutput = BertOutput(d_model,d_model)
            self.sema_module = SEMAModules(
                adapt_start_layer=0,
                adapt_end_layer=11,
                d_model=d_model,
                layer_id=1, 
                writer=None
            )


        if model_type == 'ppi_grn_mode':
            self.singlecell_model = gene_gene_operator(input_genes = self.num_genes,
                                                         grn_emb_list = self.grn_node2vec_embedding_dict, 
                                                         ppi_emb_list =self.ppi_node2vec_embedding_dict, 
                                                         device = self.device,
                                                       coexpress_network = self.coexpress_network,gene_emb_adapt = self.gene_emb_adapt,
                                                         output_dim=hidden_size)
            self.pretrained = True
        else:
            self.pretrained = False
            print('No Single cell model load!')
        if self.gene_emb_adapt:
                # self.freeze_singlecell_model()
                self.adapt_start_layer = adapt_start_layer
                self.adapt_end_layer = adapt_end_layer
                self.buffer_size = buffer_size
                self.sema_modules = nn.ModuleList()
                hidden_list = [256+d_model,d_model]
                for layer_id, h in enumerate(hidden_list):  
                    sema = SEMAModules(
                        adapt_start_layer=adapt_start_layer,  
                        adapt_end_layer=adapt_end_layer,     
                        d_model=h,  
                        layer_id=layer_id, 
                        writer=None  
                    ).to(self.device)  #
                    self.sema_modules.append(sema)  # 
    def reset_newly_added_status(self):
        self.newly_added = False
        for adapter in self.adapters:
            adapter.newly_added = False

    def freeze_functional(self):
        adapter_ls = self.adapters
        for adapter in adapter_ls:
            for param in adapter.functional.parameters():
                param.requires_grad = False
                param._grad = None
        if self.new_router is not None:
            self.fix_router()
        for param in self.router.parameters():
            param.requires_grad = False
            param._grad = None
    def freeze_singlecell_model(self):
        if hasattr(self, 'singlecell_model') and self.singlecell_model is not None:
            for param in self.singlecell_model.parameters():
                param.requires_grad = False
            print("all singlecell_model parameters is freezed")
    def freeze_rd(self):
        adapter_ls = self.adapters
        for adapter in adapter_ls:
            if adapter.rd is not None:
                for param in adapter.rd.parameters():
                    param.requires_grad = False
                    param._grad = None
                adapter.rd_loss_record.updating = False
    def freeze_pretrained_modules(self):
        for param in self.singlecell_model.parameters():
                param.requires_grad = True  
        for param in self.bert.parameters():
                param.requires_grad = False     

        # for name, param in self.bert.bert.encoder.layer.named_parameters():
        #     if 'sema_module' not in name:
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        # for param in self.bert.bert.encoder.layer[11].parameters():
        #     param.requires_grad = True  
        for param in self.encoder.parameters():
                param.requires_grad = True    
        for param in self.decoder.parameters():
                param.requires_grad = True    
        for param in self.downdecoder.parameters():
                param.requires_grad = True  
        # for param in self.intermediate.parameters():
        #         param.requires_grad = True  
        # for param in self.bertOutput.parameters():
        #         param.requires_grad = True  
        # for param in self.sema_module.parameters():
        #         param.requires_grad = True  
#         for param in self.dag.parameters():
#                 param.requires_grad = False  
#         for module in [self.bn_emb, self.emb_trans,self.comb_encoder]:
#             if module is not None:
#                 for param in module.parameters():
#                     param.requires_grad = False

#         # if hasattr(self, 'comb_encoder'):
#         #     for param in self.comb_encoder.parameters():
#         #         param.requires_grad = True  
#         for sema in self.dag.singlecell_model.sema_modules:
#             for param in sema.parameters():
#                 param.requires_grad = True  
#         for sema in self.sema_modules:
#             for param in sema.parameters():
#                 param.requires_grad = True  
#         trainable_modules = [
#             self.recovery_w, 
#             self.pert_base_trans, self.bn_pert_base_trans
#         ]
#         for module in trainable_modules:
#             if module is not None:
#                 for param in module.parameters():
#                     param.requires_grad = False

    def forward(self, x_token: torch.Tensor,x_indices: torch.Tensor,x_raw: torch.Tensor,x_value: torch.Tensor, c: torch.Tensor, noise: torch.Tensor):
        batch_size = x_token.size(0)
        species = torch.zeros(batch_size, 1, dtype=torch.int).cuda()

        forward_dict = self.bert.bert.forward(input_ids=x_token, values=x_value, species=species)
        new_emb = forward_dict[0]
        new_emb = new_emb[:,1:,:]
        
        # new_emb = self.linear_1(new_emb)
        b,n = x_raw.shape
        d,L = self.hidden_size//2,self.top_seq_num
        num_graphs = x_raw.shape[0]#number of graphs
        pert_embeddings = self.comb_encoder(c)#[b,d]
        if self.pretrained:
            emb = self.singlecell_model(new_emb,x_raw,x_indices)["merged_encodings"] #B, N, D
            emb = emb.reshape((num_graphs * self.num_genes, -1))
        else:
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.device))        
        emb = self.bn_emb(emb) #B*N, D
        base_emb = self.emb_trans(emb)  #B*N, D
        #base_emb is the basic embbeding for the genes
        #and then the drug pertubation embedding is extracted as follow:
        # out = self.recovery_w(base_emb) #B *N, D
        out = base_emb.reshape(num_graphs, self.num_genes, -1) #B, N, D
        # out = out.unsqueeze(-1) f* self.indv_w1 #B, N, D, 1
        # out = torch.sum(out, axis = 2) ##B, N
        # out = self.bn_pert_base_trans(out)#bn
        # out = self.pert_base_trans(out)#relu
        # out = out + x
        pert_emb = pert_embeddings.unsqueeze(1)
        pert_emb_repeated = pert_emb.repeat(1, 2048, 1)
        out = torch.cat((out, pert_emb_repeated), dim=2)
        out = self.downdecoder(out).squeeze(2)
        
        # out = torch.cat((out),dim=1)
        ## uncertainty head
        out = self.encoder(out)

        # out = self.bn_pert_base_trans(out)#bn
        # out = self.pert_base_trans(out)#relu
        
        if self.adapt_same:
            sema_out = self.sema_module(out.unsqueeze(1))
            
            attention_output = sema_out["func_out"].squeeze(1)
            intermediate_output = self.intermediate(attention_output)
            rd_loss = sema_out["rd_loss"]
            out = self.bertOutput(intermediate_output, attention_output)
        pert_latent = out
        out = torch.cat((out,pert_embeddings,noise),dim=1)
        out = self.decoder(out)
        dim = (out.shape[1])//3

        out = torch.cat((self.relu(out[:,:dim]), out[:, dim:]), dim=1)
        # rd_loss = forward_dict['rd_loss']+rd_loss
        return out,pert_latent #B, N
    def get_latent(self, x_token: torch.Tensor,x_indices: torch.Tensor,x_raw: torch.Tensor,x_value: torch.Tensor, c: torch.Tensor, noise: torch.Tensor):
        batch_size = x_token.size(0)
        species = torch.zeros(batch_size, 1, dtype=torch.int).cuda()

        forward_dict = self.bert.bert.forward(input_ids=x_token, values=x_value, species=species)
        new_emb = forward_dict[0]
        new_emb = new_emb[:,1:,:]
        
        # new_emb = self.linear_1(new_emb)
        b,n = x_raw.shape
        d,L = self.hidden_size//2,self.top_seq_num
        num_graphs = x_raw.shape[0]#number of graphs
        pert_embeddings = self.comb_encoder(c)#[b,d]
        if self.pretrained:
            emb = self.singlecell_model(new_emb,x_raw,x_indices)["merged_encodings"] #B, N, D
            emb = emb.reshape((num_graphs * self.num_genes, -1))
        else:
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.device))        
        emb = self.bn_emb(emb) #B*N, D
        base_emb = self.emb_trans(emb)  #B*N, D
        #base_emb is the basic embbeding for the genes
        #and then the drug pertubation embedding is extracted as follow:
        # out = self.recovery_w(base_emb) #B *N, D
        out = base_emb.reshape(num_graphs, self.num_genes, -1) #B, N, D
        # out = out.unsqueeze(-1) f* self.indv_w1 #B, N, D, 1
        # out = torch.sum(out, axis = 2) ##B, N
        # out = self.bn_pert_base_trans(out)#bn
        # out = self.pert_base_trans(out)#relu
        # out = out + x
        pert_emb = pert_embeddings.unsqueeze(1)
        pert_emb_repeated = pert_emb.repeat(1, 2048, 1)
        out = torch.cat((out, pert_emb_repeated), dim=2)
        out = self.downdecoder(out).squeeze(2)
        
        # out = torch.cat((out),dim=1)
        ## uncertainty head
        out = self.encoder(out)

        # out = self.bn_pert_base_trans(out)
        # out = self.bn_pert_base_trans(out)#bn
        # out = self.pert_base_trans(out)#relu
        # out = self.decoder(torch.cat([out,pert_embeddings,noise],dim=1))
        return pert_latent,out #B, N
def vae_loss(recon_x, x, mu, logvar,reconstruction_function,weight=1):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD * weight
    
class Pretrain_DGFM(torch.nn.Module):
    def __init__(self, 
                 num_genes=17911,
                 uncertainty=True,
                 num_gnn_layers=None,
                 decoder_hidden_size=None,
                 num_gene_gnn_layers=None,
                 input_genes_ens_ids=None,
                 scfm_genes_ens_ids=None,
                 drug_dim=None,
                 drug_gene_attr =False,
                 n_size = 20,
                 
                 coexpress_network=None,
                 gene_emb_dim=128,
                 dropout_rate=0.3,
                 hidden_size = 200,
                 pos_emb_graph = 'co_expression',
                 grn_node2vec_file='/home/MBDAI206AA201/jupyter/yhz/sc/scdata/GeneCompass-main/downstream_tasks/PRCEdrug/GraphEmbedding/node2vec/emb_grn/grn_emb_total.pkl',
                 ppi_node2vec_file='/home/MBDAI206AA201/jupyter/yhz/sc/scdata/GeneCompass-main/downstream_tasks/PRCEdrug/GraphEmbedding/node2vec/emb_ppi/ppi_emb_total.pkl',
                 model_type = 'ppi_grn_mode'
                ):
        super(Pretrain_DGFM, self).__init__()
        # self.args = args 
#G=11350++978,N=256
        self.device='cuda'
        self.num_genes = num_genes
        self.hidden_size = hidden_size
        self.uncertainty = uncertainty
        self.num_layers = num_gnn_layers
        self.indv_out_hidden_size = decoder_hidden_size
        self.num_layers_gene_pos = num_gene_gnn_layers
        self.pert_emb_lambda = 0.2
        self.input_genes_list = input_genes_ens_ids
        self.scfm_genes_list = scfm_genes_ens_ids
        self.gene_emb_adapt = True

        self.coexpress_network = coexpress_network
        try:
            with open(grn_node2vec_file, 'rb') as handle:
                self.grn_node2vec_embedding_dict = pickle.load(handle)
        except ValueError:
            import pickle5
            with open(grn_node2vec_file, 'rb') as handle:
                self.grn_node2vec_embedding = pickle5.load(handle)
        try:
            with open(ppi_node2vec_file, 'rb') as handle:
                self.ppi_node2vec_embedding_dict = pickle.load(handle)
        except ValueError:
            import pickle5
            with open(ppi_node2vec_file, 'rb') as handle:
                self.ppi_node2vec_embedding_dict = pickle5.load(handle)
        

        self.top_seq_num = 5000   

        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        dr_rate = 0.3

        self.drug_gene_attr = drug_gene_attr

        # decoder shared MLP
        # self.recovery_w = MLP([gene_emb_dim, hidden_size*2, hidden_size], last_layer_act='linear')
        

        self.relu = nn.ReLU()

        self.bn_emb = nn.BatchNorm1d(gene_emb_dim)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
              

        self.dn_linear = nn.Sequential()
        self.dn_linear.add_module(name="Le0", module=nn.Linear(self.num_genes, hidden_size*2, bias=False))
        self.dn_linear.add_module(name="B0", module=nn.BatchNorm1d(hidden_size*2))
        self.dn_linear.add_module(name="D0", module=nn.Dropout(0.3))
        self.dn_linear.add_module(name="L0", module=nn.LeakyReLU())
        self.dn_linear.add_module(name="Le1", module=nn.Linear(hidden_size*2, hidden_size))
        self.dn_linear.add_module(name="B1", module=nn.BatchNorm1d(hidden_size))
        self.dn_linear.add_module(name="D1", module=nn.Dropout(0.3))
        # self.decoder = nn.Linear(hidden_size+drug_dim+10, self.num_genes*2)
        # self.decoder = nn.Sequential()
        # self.decoder.add_module(name="Ld0", module=nn.Linear(gene_emb_dim, hidden_size, bias=False))
        # self.decoder.add_module("Nd0", module=nn.BatchNorm1d(hidden_size))
        # self.decoder.add_module(name="Ad0", module=nn.LeakyReLU(negative_slope=0.3))
        # self.decoder.add_module(name="Dd0", module=nn.Dropout(p=dropout_rate))
        # self.decoder.add_module(name="Ld1", module=nn.Linear(hidden_size, self.num_genes)) 
        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(gene_emb_dim, hidden_size*2)
        modules.append(
                nn.Sequential(
                    nn.Linear(hidden_size*2,
                                      hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.3),
                    nn.LeakyReLU()
                    ))


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_size,
                                       self.num_genes)
                            ) 
        self.fc_mu = nn.Linear(hidden_size, self.num_genes)
        self.fc_var = nn.Linear(hidden_size, self.num_genes)


        if model_type == 'ppi_grn_mode':
            self.singlecell_model = gene_gene_operator(input_genes = self.num_genes,
                                                         grn_emb_list = self.grn_node2vec_embedding_dict, 
                                                         ppi_emb_list =self.ppi_node2vec_embedding_dict, 
                                                         device = self.device,
                                                       coexpress_network = self.coexpress_network,
                                                       gene_emb_adapt = self.gene_emb_adapt,
                                                         output_dim=hidden_size)
            self.pretrained = True
        else:
            self.pretrained = False
            print('No Single cell model load!')
    def encoder(self, x: torch.Tensor):
        b,n = x.shape
        num_graphs = x.shape[0]#number of graphs
        ## get base gene embeddings
        singlecell_model = self.singlecell_model(x) #B, N, D
        emb = singlecell_model["merged_encodings"]
        emb = emb.reshape((num_graphs * self.num_genes, -1))     
        emb = self.bn_emb(emb) #
        base_emb = self.emb_trans(emb)  
        out = self.recovery_w(base_emb) #B * N, D
        out = out.reshape(num_graphs, self.num_genes, -1) #B, N, D
        out = torch.sum(out, axis = 1) ##B, N
        out = self.dn_linear(x)#B, H
        return out,singlecell_model["rd_loss"]
    def _encode(self, x: torch.Tensor):
        out,rd_loss = self.encoder(x)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        return [out,mu, log_var]

    def decode(self, z: torch.Tensor):
        """
        Maps the given latent codes
        """
        result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


    
    def encode(self, x: Tensor,repram=False):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        out, mu, log_var = self._encode(x)

        if (repram==True):
            z = self.reparameterize(mu, log_var)
            return z
        else:
            return mu

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor, **kwargs):
        out, mu, log_var = self._encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), x, mu, log_var]
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
         M_N = self.params['batch_size']/ self.num_train_imgs,
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] 
        # Account for the minibatch samples from the dataset
        # M_N = self.params['batch_size']/ self.num_train_imgs,
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples
def train_VAE_model(model,data_loaders={},optimizer=None,n_epochs=100,scheduler=None,load=False,save_path="model_VAE.pkl",best_model_cache = "drive"):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            model.load_state_dict(torch.load(save_path))           
            return model, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    dataset_sizes = {x: len(data_loaders[x].dataset) for x in ['train', 'val']}
    loss_train = {}
    
    if best_model_cache == "memory":
        best_model_wts = copy.deepcopy(model.state_dict())
    else:
        torch.save(model.state_dict(), save_path+"_bestcahce.pkl")
        model.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
    best_loss = np.inf
    for epoch in range(n_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])
            # Iterate over data.
            # for data in data_loaders[phase]:
            # Iterate over data.
            # for data in data_loaders[phase]:
            loop = tqdm(enumerate(data_loaders[phase]), total =n_iters)
            for batchidx, data in loop:
                x=data['features'][0].to('cuda')
                x.requires_grad_(True)
                # encode and decode 
                output = model(x)
                # compute loss
                recon_loss = nn.MSELoss(reduction="mean")

                loss = vae_loss(output[0],output[1],output[2],output[3],recon_loss,data_loaders[phase].batch_size/dataset_sizes[phase])

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss = running_loss+loss.item()
                loop.set_description(f'Epoch [{epoch}/{n_epochs}] [{batchidx}/{n_iters}]')
                    #loop.set_postfix(Loss_NB=nb_loss.item(), Loss_MSE=mse_loss.item())
                loop.set_postfix(Loss=loss.item())
                
            epoch_loss = running_loss / n_iters
            #epoch_loss = running_loss / n_iters
            
            if phase == 'val':
                scheduler.step()
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            print('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss

                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    torch.save(model.state_dict(), save_path+"_bestcahce.pkl")
    
    # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        model.load_state_dict(best_model_wts)  
    else:
        model.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(model.state_dict(), save_path)

    return model, loss_train