import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import random
import math
from trainer.modeling_scMCP import create_scMCP
import numpy as np
import torch
import torch.nn as nn
from trainer.loss import mmd_loss
from torch.cuda.amp import GradScaler
from torch.distributions import Bernoulli, Normal
from torch.autograd import Variable, grad
from trainer.optim import create_optimizer
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import pyarrow.parquet as pq
import pyarrow as pa
from trainer._utils import init_distributed_mode
from tqdm import tqdm
import trainer._utils as utils
import pickle
from trainer._utils import NativeScalerWithGradNormCount as NativeScaler
from data._utils import load_smiles_from_path_embedding_drug_dose_encoder
import pyarrow.parquet as pq
pf = pq.ParquetFile("shard_00031.parquet")
def dist_loss(x,y,GAMMA=1000):
        result = mmd_loss(x,y,GAMMA)
        return result
with open('gene_list.txt', 'r') as f:
    gene_list = [line.strip() for line in f if line.strip()]
# ---------------------------
class ParquetShardIterableDataset(IterableDataset):
    def __init__(self, shard_dir, split="train", rank=0, world_size=1, shuffle_shards=True, seed=0):
        super().__init__()
        self.shard_dir = Path(shard_dir) 
        self.rank = rank
        self.world_size = world_size
        self.shuffle_shards = shuffle_shards
        self.seed = seed

        # list parquet files
        files = sorted([str(p) for p in self.shard_dir.glob("*.parquet")])
        if len(files) == 0:
            raise RuntimeError(f"No parquet files found under {self.shard_dir}")
        # Shard assignment across processes (round-robin)
        # We'll produce an ordered list but each worker picks its own subset to stream
        self.files = files
    def _files_for_worker(self):
        """Return the list of files this process should iterate."""
        files = list(self.files)
        if self.shuffle_shards:
            rnd = random.Random(self.seed)
            rnd.shuffle(files)
        # split by rank using simple round-robin partition
        assigned = [f for i, f in enumerate(files) if (i % self.world_size) == self.rank]
        return assigned
    @staticmethod
    def arrow_list_to_numpy(x, dtype=np.float32):
        if hasattr(x, "values"):          # FixedSizeListScalar
            # .values is pyarrow.Array, directly transfered to numpy
            return x.values.to_numpy().astype(dtype)
        #  ListScalar
        if hasattr(x, "to_numpy"):
            return x.to_numpy().astype(dtype)
        # 
        return np.array(x, dtype=dtype)

        def _files_for_worker(self):
            """Return the list of files this process should iterate."""
            files = list(self.files)
            if self.shuffle_shards:
                rnd = random.Random(self.seed)
                rnd.shuffle(files)
            # split by rank using simple round-robin partition
            assigned = [f for i, f in enumerate(files) if (i % self.world_size) == self.rank]
            return assigned
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        files = self._files_for_worker()
        files_for_worker = [f for i, f in enumerate(files) if (i % num_workers) == worker_id]

        for file in files_for_worker:
            try:
                pf = pq.ParquetFile(file)
                for rg in range(pf.num_row_groups):
                    batch_table = pf.read_row_group(rg)

                    target_col = batch_table.column("target")
                    control_col = batch_table.column("control")
                    smiles_col = batch_table.column("smiles")
                    cov_col = batch_table.column("cov")

                    for i in range(batch_table.num_rows):
                        tgt = self.arrow_list_to_numpy(target_col[i]).reshape(-1)
                        ctrl =  self.arrow_list_to_numpy(control_col[i]).reshape(-1)

                        smile = smiles_col[i]
                        cov = cov_col[i]
                        yield {
                            "target": tgt,                     
                            "control": ctrl,
                            "smiles": smile.as_py(),         
                            "cov": cov.as_py(),               
                        }

            except Exception as e:
                print(f"Warning: failed to read {file}: {e}")
                continue

def build_encode_drug_doses(dataset="sciplex",
                            doses=[0, 50.0, 500.0, 5000.0],
                            num_bits=768):

    if dataset == "sciplex":
        file_path = "sciplex_smiles_embeddings.pkl"
    elif dataset == "tagoe":
        file_path = "Tagoe_smiles_embeddings.pkl"
    else:
        raise ValueError("dataset must be sciplex or tagoe")

    print(f"Loading SMILES embeddings from {file_path} ...")
    with open(file_path, "rb") as f:
        smiles_emb_dict = pickle.load(f)
    with open('tahoe_drug_smiles_map.pickle', "rb") as f:
        drug_smiles_map = pickle.load(f)
    drug_smiles_map = {v:k for k,v in drug_smiles_map.items()}

    smiles_list = list(smiles_emb_dict.keys())

    encode_drug_doses = {}

    print(f"Building encode_drug_doses for {len(smiles_list)} drugs × {len(doses)} doses ...")

    for smiles in tqdm(smiles_list):

        base_emb = smiles_emb_dict[smiles]  # (768,)

        for d in doses:
            # dose scaling
            scale = np.log10(d + 1)
            emb = base_emb * scale
            drug = drug_smiles_map[smiles]
            encode_drug_doses[(drug, d)] = emb.astype(np.float32)

    print(f"✔ encode_drug_doses built: total = {len(encode_drug_doses)} entries")
    print(f"Keys format: (smiles, dose) → 768-d vector")

    return encode_drug_doses
encode_drug_doses = build_encode_drug_doses(dataset="tagoe")
file = open('human_mouse_tokens.pickle', 'rb')
id_token = pickle.load(file)
file.close()
import functools

file = open('Gene_id_name_dict.pickle', 'rb')
gene = pickle.load(file)
file.close()
name2id = {value:key for key,value in gene.items()}
# ---------------------------
#  collate_fn for DataLoader: combine list of samples -> tensors
# ---------------------------
def collate_fn(samples):

    B = len(samples)

    targets = torch.tensor(
        np.stack([s["target"].squeeze() for s in samples], axis=0),
        dtype=torch.float32
    )   # (B, n_genes)

    controls = torch.tensor(
        np.stack([s["control"].squeeze() for s in samples], axis=0),
        dtype=torch.float32
    )

    drugs = []
    doses = []

    for s in samples:
        cov = s["cov"]          
        cell_,drug, dose = cov.split("_")
        drugs.append(drug)
        doses.append(float(dose))

    # encode_drug_doses: dict[(drug,dose)] = 768 vector
    labels = []
    for drug, dose in zip(drugs, doses):
        emb = encode_drug_doses[(drug, dose)]   # 768-d
        labels.append(emb)

    labels = torch.tensor(np.stack(labels, axis=0), dtype=torch.float32) # (B,768)

    top_gene_tokens_list = []
    top_gene_ids_list = []
    top_gene_idx_list = []
    controls_list = []
    top_gene_per_cell=2048
    for b in range(B):
        expr = controls[b]               # (n_genes,)
        top_vals, top_idx = torch.topk(expr, top_gene_per_cell, dim=0)

        # index → gene_id
        gene_ids = [gene_list[i.item()] for i in top_idx]

        # geneID → token
        gene_tokens = [id_token.get(name2id.get(gid,None),0) for gid in gene_ids]
        top_gene_tokens_list.append(gene_tokens)
        top_gene_ids_list.append(gene_ids)
        top_gene_idx_list.append(top_idx)
        controls_list.append(top_vals)
    controls_top = torch.tensor(torch.stack(controls_list), dtype=torch.float32) 
    top_gene_tokens = torch.tensor(top_gene_tokens_list, dtype=torch.long) # (B,K)
    top_gene_indices = torch.stack(top_gene_idx_list, dim=0)               # (B,K)

    batch = {
        "features": (controls_top, targets),
        "label": labels,  
        "cov_drug": drugs,
        "cov_dose": doses,
        "top_gene_tokens": top_gene_tokens,
        "top_gene_indices": top_gene_indices,
        "top_gene_ids": top_gene_ids_list,
        "control_raw": controls,
    }

    return batch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# ---------------------------
from trainer.engine_pretraining import train_one_epoch,evalute  

def get_args():
    parser = argparse.ArgumentParser("DGFM streaming training")
    parser.add_argument("--shard_root", default="/raid/MBDAI/tahoe", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--num_shards_read", default=None, type=int)
    parser.add_argument("--shuffle_shards",default=True, action="store_true")
    parser.add_argument("--save_dir", default="./ckpt", type=str)
    parser.add_argument("--distributed", default=True, action="store_true")
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument("--gpu", default=7, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--world_size", default=7, type=int, help="number of distributed processes"
    )
    
    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=1e-8,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""",
    )

    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=1,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")

    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        "--start_epoch", default=1, type=int, metavar="N", help="start epoch"
    )

    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    return parser.parse_args()

def main():
    args = get_args()
    init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    dataset = ParquetShardIterableDataset(
        shard_dir=args.shard_root,
        split=args.split,
        rank=sampler_rank,
        world_size=num_tasks,
        shuffle_shards=args.shuffle_shards,
        seed=args.seed,
    )

    # For DDP + IterableDataset, use DataLoader with shuffle=False and no DistributedSampler.
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
    )
    test_dataset = ParquetShardIterableDataset(
        shard_dir=args.shard_root+'/test001',
        split=args.split,
        rank=sampler_rank,
        world_size=num_tasks,
        shuffle_shards=args.shuffle_shards,
        seed=args.seed,
    )
     
    test_dataloader= DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
    )
    import glob
    def count_samples_for_rank(shard_root, world_size, rank):
        files = sorted(glob.glob(os.path.join(shard_root, "*.parquet")))
        assigned = files[rank::world_size]
        total = 0
        for f in assigned:
            total += pq.ParquetFile(f).metadata.num_rows
        return total

    train_samples = count_samples_for_rank(
        args.shard_root,
        world_size=num_tasks,
        rank=sampler_rank
    )
    
    num_training_steps_per_epoch = train_samples // args.batch_size
    print("steps per epoch: ", num_training_steps_per_epoch)
    lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=0)
    wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs,
            num_training_steps_per_epoch)
    print("DEBUG: rank", sampler_rank, "world_size", num_tasks)


    model = create_scMCP(pretrained=False)
    model.to(device)

    if True:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module    
    # optimizer & scaler
    optimizer = create_optimizer(args, model_without_ddp)
    # loss_scaler = NativeScaler()
    AMP_DTYPE = "float16"
    loss_scaler = GradScaler(enabled=(AMP_DTYPE == 'float16'))

    def make_noise(device, batch_size, shape, volatile=False):
        tensor = torch.randn(batch_size, shape)
        noise = Variable(tensor, volatile)
        noise = noise.to(device, dtype=torch.float32)
        return noise
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



    # ---------------------------
    #  Training loop (epoch)
    # ---------------------------
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # If distributed, no need to set sampler epoch for IterableDataset; but we can reseed
        dataset.seed = args.seed + epoch  # if dataset uses seed
        model.train()
        train_stats = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=loss_scaler,
            device=device,
            epoch=epoch,
            loss_fn=ZILNCriterion(),
            dist_loss=dist_loss,
            make_noise_fn=make_noise,
            sample_ziln_fn=sample_ziln,
            GRAD_ACCUM=args.grad_accum,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            start_steps=(epoch - 1) * num_training_steps_per_epoch,
                   lr_schedule_values=lr_schedule_values,
        wd_schedule_values=wd_schedule_values,
            log_writer=None,
            args=args
        )
        eval_stats = evalute(
            model=model,
            dataloader=test_dataloader,
            optimizer=optimizer,
            scaler=loss_scaler,
            device=device,
            epoch=epoch,
            loss_fn=ZILNCriterion(),
            dist_loss=dist_loss,
            make_noise_fn=make_noise,
            sample_ziln_fn=sample_ziln,
            GRAD_ACCUM=args.grad_accum,
            log_writer=None,
            
            args=args
        )
        epoch_time = time.time() - epoch_start
        if sampler_rank == 0:
            print(f"Epoch {epoch} done in {epoch_time/60:.2f} min. Metrics: {train_stats}")
            print(f"Epoch {epoch} done in {epoch_time/60:.2f} min. Metrics: {eval_stats}")
            # simple checkpoint
            ckpt_dir = Path(args.save_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict() if not args.distributed else model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_dir / f"checkpoint_epoch_{epoch}.pt")

    total_time = time.time() - start_time
    if sampler_rank == 0:
        print("Total training time:", str(datetime.timedelta(seconds=int(total_time))))

if __name__ == "__main__":
    main()
