# -*- coding: utf-8 -*-
# @Author: Hongzhi Yao
# @Date:   2025-06-21 09:15:55
# @Last Modified by:   Hongzhi Yao
# @Last Modified time: 2025-7-5 18:33:14
import os
import shutil
import sys
from functools import partial
import argparse
import torch
import random
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, load_from_disk
import yaml
import getpass
import numpy as np
from datetime import datetime
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.distributed as dist
import pytorch_lightning as pl
import os
from ginet_3emb_degree import GNNDecoder, GNN
from loss import NTXentLoss, sce_loss
from torch.cuda.amp import autocast, GradScaler
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
from pytorch_lightning.callbacks import LearningRateMonitor
apex_support = False
# from apex import optimizers
from loader import MoleculeProcessor
from torch.distributed import broadcast_object_list
import subprocess
import glob
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False
class CheckpointEveryNSteps(pl.Callback):
    """
        Save a checkpoint every N steps, instead of Lightning's default that checkpoints
        based on validation loss.
    """

    def __init__(self, save_step_frequency=-1,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
        ):
        """
        Args:
        save_step_frequency: how often to save in steps
        prefix: add a prefix to the name, only used if
        use_modelcheckpoint_filename=False
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0 and self.save_step_frequency > 10:

            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)
@rank_zero_only
def remove_tree(cachefiles):
    if type(cachefiles) == type([]):
        #if cachefiles are identical remove all but one file path
        cachefiles = list(set(cachefiles))
        for cache in cachefiles:
            shutil.rmtree(cache)
    else:
        shutil.rmtree(cachefiles)

def random_remask(dec_mask_token, rep, x, device, remask_rate=0.5):
    num_nodes = x.num_nodes
    perm = torch.randperm(num_nodes,device = device)
    num_remask_nodes = int(remask_rate * num_nodes)
    remask_nodes = perm[:num_remask_nodes]
    
    rep_new = torch.zeros_like(rep)
    rep_new.copy_(rep)
    
    mask = torch.zeros_like(rep_new, dtype=torch.bool)
    mask[remask_nodes] = True
    
    rep_new = torch.where(
        mask,
        dec_mask_token.expand_as(rep_new),  
        rep_new
    )
    
    return rep_new, remask_nodes, None

class MoleculeModule(pl.LightningDataModule):
    def __init__(self,  max_len, data_path, train_args):
        super().__init__()
        self.data_path = ['pubchem']
        self.train_args = train_args  # dict with keys {'batch_size', 'shuffle', 'num_workers', 'pin_memory'}
        print(train_args)
        self.data_collector = MoleculeProcessor()
        
        
    def prepare_data(self):
        pass

    def get_cache(self):
        return self.cache_files
    def setup(self, stage=None):
        #######################Pubchem dataset
        # pubchem_path = {'train':'/home/MBDAI206AA201/jupyter/yhz/sc/molformer-main/data/pubchem-canonical/CID-SMILES-CANONICAL.smi'}
        # pubchem_script = '/home/MBDAI206AA201/jupyter/yhz/sc/molformer-main/training/pubchem_script.py'
        # dataset_dict =  load_dataset(pubchem_script, data_files=pubchem_path, cache_dir=os.path.join('/home/MBDAI206AA201/jupyter/yhz/tmp',getpass.getuser(), 'pubchem'), split='train')    
        #######################ZINC dataset
        zinc_path = '/home/MBDAI206AA201/jupyter/yhz/sc/molformer-main/data/ZINC'
        zinc_files = [f for f in glob.glob(os.path.join(zinc_path,'*.smi'))][:110]
        for zfile in zinc_files:
            print(zfile)
        self.data_path = {'train': zinc_files}
        dataset_dict = load_dataset('./zinc_script.py', data_files=self.data_path, cache_dir=os.path.join('/home/MBDAI206AA201/jupyter/yhz/tmp',getpass.getuser(), 'zinc'),split='train')
        self.pubchem= dataset_dict
        self.cache_files = []
        for cache in dataset_dict.cache_files:
            tmp = '/'.join(cache['filename'].split('/')[:4])
            self.cache_files.append(tmp)
    def train_dataloader(self):
        loader =  DataLoader(self.pubchem, collate_fn=self.data_collector.process, **self.train_args)
        print(len(loader))
        return loader

    def val_dataloader(self):
        return []
    def test_dataloader(self):
        return []
class MolGATMAE(pl.LightningModule):
    
    def __init__(self,  config):
        super().__init__() 
        self.config = config
        self._num_remasking =2
        self._remask_rate=0.3
        self.loss_fn="sce"
        NUM_NODE_ATTR = 119+4+7+12+10+12+8
        NUM_BOND_ATTR = 5 + 3 + 3 + 3
        self.cur_device = 'cuda' 
        self.encoder = GNN(num_layer=self.config['num_layer'], emb_dim=self.config['emb_dim'],
                    JK=self.config['JK'],feat_dim=self.config['feat_dim'], drop_ratio=self.config['dropout_ratio'],
                    gnn_type='degree',degree_list=[0, 1, 2, 3, 4, 5 ,6 ,7, 8, 9, 10],
                    batch_size=self.config['batch_size'],device=self.cur_device).double()
        # self.encoder = GNN(num_layer=self.config['num_layer'], emb_dim=self.config['emb_dim'],
        #             JK=self.config['JK'],feat_dim=self.config['feat_dim'], drop_ratio=self.config['dropout_ratio'],
        #             gnn_type='degree',degree_list=[0, 1, 2, 3, 4, 5 ,6 ,7, 8, 9, 10],
        #             batch_size=self.config['batch_size'],device=self.cur_device).double()
        if self.config['input_model_file'] is not None and self.config['input_model_file'] != "":
            model.load_state_dict(torch.load(self.config['input_model_file']))
            print("Resume training from:", self.config['input_model_file'])
            self.resume = True
        else:
            self.resume = False
        self.dec_pred_atoms = GNNDecoder(self.config['emb_dim'], NUM_NODE_ATTR, JK=self.config['JK'], gnn_type=self.config['gnn_type']).double()
        self.dec_pred_bonds = GNNDecoder(self.config['emb_dim'], NUM_BOND_ATTR, JK=self.config['JK'], gnn_type='linear').double()
            
        self.nt_xent_criterion = NTXentLoss(self.cur_device, config['batch_size'], config['temperature'], config['use_cosine_similarity']).double()
        alpha_l=1.0
        self.criterion = partial(sce_loss, alpha=alpha_l)
    def on_save_checkpoint(self, checkpoint):
        #save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state']=torch.get_rng_state()
        out_dict['cuda_state']=torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state']=np.random.get_state()
        if random:
            out_dict['python_state']=random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        #load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key =='torch_state':
                torch.set_rng_state(value)
            elif key =='cuda_state':
                torch.cuda.set_rng_state(value)
            elif key =='numpy_state':
                np.random.set_state(value)
            elif key =='python_state':
                random.setstate(value)
            else:
                print('unrecognized state')
    def on_validation_epoch_end(self, outputs):

        avg_loss = torch.tensor([output['loss'] for output in outputs]).mean()
        loss = {'loss': avg_loss.item()}
        self.log('validation_loss', loss['loss'])
    def validation_step(self, batch, batch_idx):
        # idxl =  batch[0]
        # loss = 0
        
        # for chunk in range(len(idxl)):
        xis, xjs=batch
        # xis = xis.to(self.device, non_blocking=True)
        # xjs = xjs.to(self.device, non_blocking=True)
        node_rep,zis = self.encoder(xis)
        #random remask
        
        node_rep_j,zjs = self.encoder(xjs
        node_rep = node_rep.clone()  
        node_rep_j = node_rep_j.clone()
        
        node_attr_label = xis.node_attr_label
        masked_node_indices = xis.masked_atom_indices
        pred_node =  self.dec_pred_atoms(node_rep, xis.edge_index, xis.edge_attr, masked_node_indices)


        node_attr_label_j = xjs.node_attr_label
        masked_node_indices_j = xjs.masked_atom_indices
        pred_node_j =  self.dec_pred_atoms(node_rep_j, xjs.edge_index, xjs.edge_attr, masked_node_indices_j)
        # loss = criterion(pred_node.double(), batch.mask_node_label[:,0])
        if self.loss_fn == "sce":
            latent_loss = self.criterion(node_attr_label, pred_node[masked_node_indices])
            latent_loss = latent_loss + self.criterion(node_attr_label_j, pred_node_j[masked_node_indices_j])
        else:
            latent_loss = self.criterion(pred_node.double()[masked_node_indices], xis.mask_node_label[:, 0])

        if self.config['mask_edge']:
            masked_edge_index = xis.edge_index[:, xis.connected_edge_indices]
            edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            pred_edge = self.dec_pred_bonds(edge_rep, xis.edge_index, xis.edge_attr, masked_node_indices)
            edge_loss =  self.criterion(pred_edge.double(), xis.edge_attr_label)

            masked_edge_index_j = xjs.edge_index[:, xjs.connected_edge_indices]
            edge_rep_j = node_rep[masked_edge_index_j[0]] + node_rep[masked_edge_index_j[1]]
            pred_edge_j = self.dec_pred_bonds(edge_rep_j, xjs.edge_index, xjs.edge_attr, masked_node_indices_j)
            edge_loss = edge_loss + self.criterion(pred_edge_j.double(), xjs.edge_attr_label)
            # acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:,0])
            # acc_edge_accum += acc_edge


        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        if(zis.shape[0]!=self.nt_xent_criterion.batch_size):
            self.nt_xent_criterion.batch_size=zis.shape[0]
            self.nt_xent_criterion.mask_samples_from_same_repr = self.nt_xent_criterion._get_correlated_mask().type(torch.bool)
        total_loss = edge_loss + latent_loss  + self.nt_xent_criterion(zis, zjs)
        # if chunk < len(idxl)-1:
        #     total_loss.backward()
        #     loss += total_loss.detach()
        # else:
        #     loss += total_loss
        return {'loss':total_loss}
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def configure_optimizers(self):
        model_params = []
        dec_pred_atoms_params = []
        dec_pred_bonds_params = []
        # print(self.named_parameters())
        for name, param in self.named_parameters():
            if 'dec_pred_atoms' in name: 
                dec_pred_atoms_params.append(param)
            elif 'dec_pred_bonds' in name: 
                dec_pred_bonds_params.append(param)
            else: 
                model_params.append(param)

     
        param_groups = [
            {
                'params': model_params,
                'lr': self.config['init_lr'],  
                'weight_decay': self.config['weight_decay']  
            },
            {
                'params': dec_pred_atoms_params,
                'lr': self.config['init_lr'],  
                'weight_decay': self.config['weight_decay'] 
            },
            {
                'params': dec_pred_bonds_params,
                'lr': self.config['init_lr'],  
                'weight_decay': self.config['weight_decay']
            }
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.99) 
        )
        return optimizer
    def training_step(self, batch, batch_idx):
        xis, xjs=batch

        node_rep,zis = self.encoder(xis)
        #random remask

        node_rep_j,zjs = self.encoder(xjs)
        
        node_rep = node_rep.clone() 
        node_rep_j = node_rep_j.clone()
        loss_rec_all = 0
        masked_node_indices_i = xis.masked_atom_indices
        masked_node_indices_j = xjs.masked_atom_indices

        for i in range(self._num_remasking):
            rep = node_rep.clone().detach().requires_grad_(True)
            rep_j = node_rep_j.clone().detach().requires_grad_(True)
            with torch.no_grad():
                rep_masked, remask_nodes, _ = random_remask(
                    self.encoder.dec_mask_token,
                    rep,
                    xis,
                   self.cur_device,
                    self._remask_rate
                )
                
                rep_j_masked, remask_nodes_j, _ = random_remask(
                    self.encoder.dec_mask_token,
                    rep_j,
                    xjs,
                   self.cur_device,
                    self._remask_rate
                )
            rep_masked = rep + (rep_masked - rep).detach()
            rep_j_masked = rep_j + (rep_j_masked - rep_j).detach()

            recon = self.dec_pred_atoms(rep_masked, xis.edge_index, xis.edge_attr, masked_node_indices_i)
            recon_j = self.dec_pred_atoms(rep_j_masked, xjs.edge_index, xjs.edge_attr, masked_node_indices_j)
            x_init = xis.node_attr_label[masked_node_indices_i]
            x_rec = recon[masked_node_indices_i]
            loss_rec_all = loss_rec_all + self.criterion(x_init, x_rec)
            x_init = xis.node_attr_label[masked_node_indices_j]
            x_rec = recon_j[masked_node_indices_j]
            loss_rec_all = loss_rec_all + self.criterion(x_init, x_rec)

        node_attr_label = xis.node_attr_label
        masked_node_indices = xis.masked_atom_indices
        pred_node =  self.dec_pred_atoms(node_rep, xis.edge_index, xis.edge_attr, masked_node_indices)


        node_attr_label_j = xjs.node_attr_label
        masked_node_indices_j = xjs.masked_atom_indices
        pred_node_j =  self.dec_pred_atoms(node_rep_j, xjs.edge_index, xjs.edge_attr, masked_node_indices_j)
        # loss = criterion(pred_node.double(), batch.mask_node_label[:,0])
        if self.loss_fn == "sce":
            latent_loss = self.criterion(node_attr_label, pred_node[masked_node_indices])
            latent_loss = latent_loss + self.criterion(node_attr_label_j, pred_node_j[masked_node_indices_j])
        else:
            latent_loss = self.criterion(pred_node.double()[masked_node_indices], xis.mask_node_label[:, 0])

        if self.config['mask_edge']:
            masked_edge_index = xis.edge_index[:, xis.connected_edge_indices]
            edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            pred_edge = self.dec_pred_bonds(edge_rep, xis.edge_index, xis.edge_attr, masked_node_indices)
            edge_loss =  self.criterion(pred_edge.double(), xis.edge_attr_label)

            masked_edge_index_j = xjs.edge_index[:, xjs.connected_edge_indices]
            edge_rep_j = node_rep[masked_edge_index_j[0]] + node_rep[masked_edge_index_j[1]]
            pred_edge_j = self.dec_pred_bonds(edge_rep_j, xjs.edge_index, xjs.edge_attr, masked_node_indices_j)
            edge_loss = edge_loss + self.criterion(pred_edge_j.double(), xjs.edge_attr_label)
            # acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:,0])
            # acc_edge_accum += acc_edge


        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        if(zis.shape[0]!=self.nt_xent_criterion.batch_size):
            self.nt_xent_criterion.batch_size=zis.shape[0]
            self.nt_xent_criterion.mask_samples_from_same_repr = self.nt_xent_criterion._get_correlated_mask().type(torch.bool)
        total_loss = edge_loss + latent_loss + loss_rec_all + self.nt_xent_criterion(zis, zjs)
        return {'loss':total_loss}



def load_smiles_from_files(filepaths):
    smiles_list = []
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            for line in file:
                smiles = line.strip().split()[0] 
                if smiles=='smiles':
                    continue
                smiles_list.append(smiles)
    return smiles_list
def get_nccl_socket_ifname():
    ipa = subprocess.run(['ip', 'a'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ipa.stdout.decode('utf-8').split('\n')
    all_names = []
    name = None
    for line in lines:
        if line and not line[0] == ' ':
            name = line.split(':')[1].strip()
            continue
        if 'link/infiniband' in line:
            all_names.append(name)
    os.environ['NCCL_SOCKET_IFNAME'] = 'en,eth,em,bond'
# ','.join(all_names)
def fix_infiniband():
    # os.environ['NCCL_SOCKET_IFNAME'] = "^lo,docker,virbr,vmnet,vboxnet,wl,ww,ppp,bond"

    # ifname = os.environ.get('NCCL_SOCKET_IFNAME', None)
    # if ifname is None:
    #     os.environ['NCCL_SOCKET_IFNAME'] = '^lo,docker0'

    get_nccl_socket_ifname()
    os.environ['NCCL_IB_CUDA_SUPPORT'] = '1'
    ibv = subprocess.run('ibv_devinfo', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ibv.stdout.decode('utf-8').split('\n')
    exclude = ''
    for line in lines:
        if 'hca_id:' in line:
            name = line.split(':')[1].strip()
        if '\tport:' in line:
            port = line.split(':')[1].strip()
        if 'link_layer:' in line and 'Ethernet' in line:
            exclude = exclude + f'{name}:{port},'
    if exclude:
        exclude = '^' + exclude[:-1]
        # print(exclude)
        os.environ['NCCL_IB_HCA'] = exclude

def main():
    
    fix_infiniband()
    config={
        'batch_size': 360,  # batch size
        'warm_up': 2,  # warm-up epochs
        'epochs': 100, # total number of epochs
'num_nodes':1,
        'load_model': None,  # resume training
        'eval_every_n_epochs': 1,  # validation frequency
        'save_every_n_epochs': 1 , # automatic model saving frequecy
        'log_every_n_steps': 1,  # print training log frequency

        'fp16_precision': False,  # float precision 16 (i.e. True/False)
        'init_lr': 3e-5*8 ,  # initial learning rate for Adam
        'weight_decay': 1e-5,  # weight decay for Adam
'world_size': 6,
    'model_type': 'gin' , # GNN backbone (i.e., gin/gcn)

    'num_layer': 6,  # number of graph conv layers
    'emb_dim': 1024,  # embedding dimension in graph conv layers
    'feat_dim': 768,  # output feature dimention
    'drop_ratio': 0,  # dropout ratio
    'pool': 'mean',  # readout pooling (i.e., mean/max/add)

    'aug': 'node',  # molecule graph augmentation strategy (i.e., node/subgraph/mix)

    'num_workers': 12,  # dataloader number of workers
    'valid_size': 0.05,  # ratio of validation data
    
    'temperature': 0.1,  # temperature of NT-Xent loss
    'use_cosine_similarity': True,  # whet
        'JK': 'last',
    'dropout_ratio': 0.0,
    'gnn_type': 'gin',
    'mask_rate':0.15,
    'mask_edge':True,
        'input_model_file':'',
        'use_scheduler':True,
        'alpha_l':1.0,
        'output_model_file':'gin_pretrain_remask_20250529_ZINC_dist',
        'loss_fn':'sce',
        'restart_path':'',
        'max_len':202
    }
    import scanpy as sc
    import getpass
    if config['num_nodes'] > 1:
        # print("Using " + str(config.num_nodes) + " Nodes----------------------------------------------------------------------")
        LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split(' ') # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
        HOST_LIST = LSB_MCPU_HOSTS[::2] # Strips the cores per node items in the list
        os.environ["MASTER_ADDR"] = HOST_LIST[0] # Sets the MasterNode to thefirst node on the list of hosts
        os.environ["MASTER_PORT"] = "54966"
        os.environ["NODE_RANK"] = str(HOST_LIST.index(os.environ["HOSTNAME"])) #Uses the list index for node rank, master node rank must be 0
        #os.environ["NCCL_SOCKET_IFNAME"] = 'ib,bond'  # avoids using docker of loopback interface
        os.environ["NCCL_DEBUG"] = "INFO" #sets NCCL debug to info, during distributed training, bugs in code show up as nccl errors
        #os.environ["NCCL_IB_CUDA_SUPPORT"] = '1' #Force use of infiniband
        #os.environ["NCCL_TOPO_DUMP_FILE"] = 'NCCL_TOP.%h.xml'
        #os.environ["NCCL_DEBUG_FILE"] = 'NCCL_DEBUG.%h.%p.txt'
        print(os.environ["HOSTNAME"] + " MASTER_ADDR: " + os.environ["MASTER_ADDR"])
        print(os.environ["HOSTNAME"] + " MASTER_PORT: " + os.environ["MASTER_PORT"])
        print(os.environ["HOSTNAME"] + " NODE_RANK " + os.environ["NODE_RANK"])
        print(os.environ["HOSTNAME"] + " NCCL_SOCKET_IFNAME: " + os.environ["NCCL_SOCKET_IFNAME"])
        print(os.environ["HOSTNAME"] + " NCCL_DEBUG: " + os.environ["NCCL_DEBUG"])
        print(os.environ["HOSTNAME"] + " NCCL_IB_CUDA_SUPPORT: " + os.environ["NCCL_IB_CUDA_SUPPORT"])
        print("Using " + str(config['num_nodes']) + " Nodes---------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    else:
        print("Using " + str(config['num_nodes']) + " Node----------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    train_config = {'batch_size':config['batch_size'], 'num_workers':config['num_workers'], 'pin_memory':True}
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
    every_n_epochs=1,  # Save checkpoint every 1 epoch
    save_top_k=-1,      # Save all checkpoints (you can change this to a positive integer to save top-k checkpoints based on a monitored metric)
    verbose=True        # Print information about checkpoint saving
)
    checkpoint_path = './lightning_logs/version_60/checkpoints/epoch=0-step=74552.ckpt'
    train_loader = MoleculeModule(config['max_len'], None, train_config)
    train_loader.setup()
    cachefiles = train_loader.get_cache()

    molgatmae = MolGATMAE(config)
    trainer = pl.Trainer(default_root_dir='.',
                max_epochs=config['epochs'],
                accelerator="gpu", strategy='ddp_find_unused_parameters_true', 
                num_nodes=1,
                devices=7,
                         
                callbacks=[checkpoint_callback,ModelCheckpointAtEpochEnd(), CheckpointEveryNSteps(1000)],
          
                accumulate_grad_batches=2,
                num_sanity_val_steps=10,
                val_check_interval=50)
    # ckpt_path=checkpoint_path, 
    try:
        trainer.fit(model=molgatmae,  train_dataloaders=train_loader)
    except Exception as exp:
        print(type(exp))
        print(exp)
        rank_zero_warn('We have caught an error, trying to shut down gracefully')
        remove_tree(cachefiles)

if __name__ == "__main__":
    main()