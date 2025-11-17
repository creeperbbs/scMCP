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

from sklearn.metrics import r2_score, mean_squared_error
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
from data.DGFM_Dataset import  AnnDataset,DrugDoseAnnTokenDataset
from models.DA import Discriminator, GeneratorDA, GeneratorAD
from trainer.loss import mmd_loss
from ._utils import train_valid_test,train_valid_test_no_dose
from .co_express_networks import get_similarity_network,GeneSimNetwork
from torch.distributions import RelaxedBernoulli
from typing import Iterable
import torch
import torch.nn as nn
from contextlib import nullcontext
import trainer._utils as utils



def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch,
    loss_fn,
    dist_loss,
    make_noise_fn,
    sample_ziln_fn,
    GRAD_ACCUM=1,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch = 0,
    log_writer=None,
    start_steps=0,
    args=None
):
    model.train()
    print(f"Epoch [{epoch}]")

    total_rec_loss = 0.0
    total_mmd_loss = 0.0
    total_r2 = 0.0
    total_steps = 0
    total_var = 0
    for step, data in tqdm(enumerate(dataloader),total=num_training_steps_per_epoch):
        it = start_steps+step
        optimizer.zero_grad(set_to_none=True)
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):

                # ---- learning rate ----
                if lr_schedule_values is not None:
                    if it < len(lr_schedule_values):     
                        new_lr = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                        param_group["lr"] = new_lr
                    # else: 
                # ---- weight decay ----
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    if it < len(wd_schedule_values):    
                        param_group["weight_decay"] = wd_schedule_values[it]
        (control, target) = data['features']
        encode_label = data['label']
        gene_token = data['top_gene_tokens']
        x_indices = data['top_gene_indices']
        x_raw = data['control_raw']

        control = control.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)
        encode_label = encode_label.to(device, dtype=torch.float32)
        x_raw = x_raw.to(device, dtype=torch.float32)
        gene_token = gene_token.to(device)
        x_indices = x_indices.to(device)

        bsz = control.size(0)
        noise = make_noise_fn(device,bsz, 20)
        control_noise = make_noise_fn(device,bsz, 20)
        ptdtype = torch.float16
        with autocast(device_type='cuda', dtype=ptdtype):

            gene_rec, latent = model(gene_token, x_indices, x_raw, control, encode_label, noise)
            # ctrl_rec, ctrl_lat = model(gene_token, x_indices, x_raw, control, encode_label * 0, control_noise)

            dim = gene_rec.size(1) // 3
            gene_mean = gene_rec[:, :dim]
            gene_var  = F.softplus(gene_rec[:, dim:2*dim])
            gene_zi   = gene_rec[:, 2*dim:]
            # ctrl_mean = ctrl_rec[:, :dim]
            # ctrl_var  = F.softplus(ctrl_rec[:, dim:2*dim])
            # ctrl_zi   = ctrl_rec[:, 2*dim:]
            gene_mean = torch.nan_to_num(gene_mean, nan=0.0, posinf=1e3, neginf=-1e3)
            gene_var = torch.clamp(gene_var, min=1e-8, max=1e4)
                        
            gene_scale = torch.sqrt(gene_var + 1e-8)
            # ctrl_scale = torch.sqrt(ctrl_var + 1e-8)

            rec_loss = (
                loss_fn(gene_mean, gene_var, gene_zi, target)
            )

            zi_prob = torch.sigmoid(gene_zi)
            log_normal_dist = Normal(
                loc=torch.clamp(gene_mean, -1e3, 1e3),
                scale=gene_scale
            )
            nb_sample = sample_ziln_fn(zi_prob, log_normal_dist, device,truncate_max=1e4 )

        

            loss = rec_loss

        # backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            nb_sample_np = nb_sample.cpu().numpy()
            yp_m = nb_sample_np.mean(0)
            yt_m = target.cpu().numpy().mean(0)
            r2 = r2_score(yp_m,yt_m)
            var_m = nb_sample_np.var(0)
            vart_m = target.cpu().numpy().var(0)
            var = r2_score(var_m,vart_m)
        # accumulate
        total_rec_loss += rec_loss.item()
        # total_mmd_loss += mmd_loss.item()
        total_r2 += r2
        total_steps += 1
        total_var += var
        print(f"Step {step:05d} | Rec={rec_loss.item():.4f} |  R2={r2:.3f}|var={var:.3f}")

    rec_avg = total_rec_loss / total_steps
    # mmd_avg = total_mmd_loss / total_steps
    r2_avg = total_r2 / total_steps
    var_avg = total_var / total_steps
    return {
        "rec_loss_avg": rec_avg,
        "total_loss_avg": rec_avg ,
        "r2_avg": r2_avg,
        "var_avg": var_avg,
    }

def evalute(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch,
    loss_fn,
    dist_loss,
    make_noise_fn,
    sample_ziln_fn,
    GRAD_ACCUM=1,
    log_writer=None,
    args=None
):
    model.train()
    print(f"Epoch [{epoch}]")

    total_rec_loss = 0.0
    total_mmd_loss = 0.0
    total_r2 = 0.0
    total_steps = 0
    for step, data in tqdm(enumerate(dataloader)):
        # optimizer.zero_grad(set_to_none=True)
        model.eval()
        (control, target) = data['features']
        encode_label = data['label']
        gene_token = data['top_gene_tokens']
        x_indices = data['top_gene_indices']
        x_raw = data['control_raw']

        control = control.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)
        encode_label = encode_label.to(device, dtype=torch.float32)
        x_raw = x_raw.to(device, dtype=torch.float32)
        gene_token = gene_token.to(device)
        x_indices = x_indices.to(device)

        bsz = control.size(0)
        noise = make_noise_fn(device,bsz, 20)
        control_noise = make_noise_fn(device,bsz, 20)
        with torch.no_grad():
            gene_rec, latent = model(gene_token, x_indices, x_raw, control, encode_label, noise)
            # ctrl_rec, ctrl_lat = model(gene_token, x_indices, x_raw, control, encode_label * 0, control_noise)

            dim = gene_rec.size(1) // 3
            gene_mean = gene_rec[:, :dim]
            gene_var  = F.softplus(gene_rec[:, dim:2*dim])
            gene_zi   = gene_rec[:, 2*dim:]
            # ctrl_mean = ctrl_rec[:, :dim]
            # ctrl_var  = F.softplus(ctrl_rec[:, dim:2*dim])
            # ctrl_zi   = ctrl_rec[:, 2*dim:]

            gene_scale = torch.sqrt(gene_var + 1e-8)
            # ctrl_scale = torch.sqrt(ctrl_var + 1e-8)

            rec_loss = (
                loss_fn(gene_mean, gene_var, gene_zi, target)
            )

            zi_prob = torch.sigmoid(gene_zi)
            log_normal_dist = Normal(
                loc=torch.clamp(gene_mean, -1e3, 1e3),
                scale=gene_scale
            )
            nb_sample = sample_ziln_fn(zi_prob, log_normal_dist, device)



        
            nb_sample_np = nb_sample.cpu().numpy()
            yp_m = nb_sample_np.mean(0)
            yt_m = target.cpu().numpy().mean(0)
            r2 = r2_score(yp_m,yt_m)

        # accumulate
        total_rec_loss += rec_loss.item()
        # total_mmd_loss += mmd_loss.item()
        total_r2 += r2
        total_steps += 1

        print(f"Step {step:05d} | Rec={rec_loss.item():.4f} |  R2={r2:.3f}")

    # mmd_avg = total_mmd_loss / total_steps
    r2_avg = total_r2 / total_steps

    return {
        "r2_avg": r2_avg,
    }
