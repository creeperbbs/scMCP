# -*- coding: utf-8 -*-
# @Author: Hoingzhi Yao
# @Date:   2025-04-05 12:55:43
# @Last Modified by:    Hoingzhi Yao
# @Last Modified time: 2025-09
import torch.distributed as dist
from cgi import test
from random import shuffle
import torch.distributed as dist
import math
import sys
import anndata
import numpy as np
from scipy import sparse
from sklearn import preprocessing
import torch
import os
from anndata import AnnData
from collections import defaultdict
def get_model(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(
        model, torch.nn.parallel.DistributedDataParallel
    ):
        return model.module
    else:
        return model
from collections import defaultdict, deque
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
import time
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
def shuffle_adata(adata):
    """
    Shuffles the `adata`.
    """
    if sparse.issparse(adata.X):
        #adata.X: sparse matrix to array
        adata.X = adata.X.A
    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    new_adata = adata[ind_list, :]
    return new_adata

def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    else:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])


def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])


def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = (
            _get_world_size_env()
        )  # int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = _get_local_rank_env()
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def train_valid_test(adata: AnnData, split_key = 'cov_drug_dose_name_split'):
    '''
    Get train_valid_test dataset
    '''


    shuffled = shuffle_adata(adata)
    train_index = adata.obs[adata.obs[split_key]=='train'].index.tolist()
    valid_index = adata.obs[adata.obs[split_key]=='valid'].index.tolist()
    test_index = adata.obs[adata.obs[split_key]=='test'].index.tolist()
    control_index = adata.obs[adata.obs['dose']==0.0].index.tolist()


    if len(train_index)>0:
        train_index = train_index + control_index
        train_adata = shuffled[train_index, :]
    else:
        train_adata = None
    if len(valid_index)>0:
        valid_index = valid_index + control_index
        valid_adata = shuffled[valid_index, :]
    else:
        valid_adata=None
    if len(test_index)>0:
        test_index = test_index + control_index
        test_adata = shuffled[test_index, :]
    else:
        test_adata=None

    
    return train_adata, valid_adata, test_adata


def train_valid_test_no_dose(adata: AnnData, split_key = 'cov_drug_dose_name_split'):
    '''
    Get train_valid_test dataset
    '''


    shuffled = shuffle_adata(adata)
    train_index = adata.obs[adata.obs[split_key]=='train'].index.tolist()
    valid_index = adata.obs[adata.obs[split_key]=='valid'].index.tolist()
    test_index = adata.obs[adata.obs[split_key]=='test'].index.tolist()


    if len(train_index)>0:
        train_adata = shuffled[train_index, :]
    else:
        train_adata = None
    if len(valid_index)>0:
        valid_adata = shuffled[valid_index, :]
    else:
        valid_adata=None
    if len(test_index)>0:
        test_adata = shuffled[test_index, :]
    else:
        test_adata=None

    
    return train_adata, valid_adata, test_adata

def train_valid(adata: AnnData, split_key=None, train_ratio=0.2, valid_ratio=0.4, test_ratio=0.4):
    '''
    Get train_valid_test dataset with optional random splitting
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    split_key : str, optional
        Key in adata.obs to use for splitting. If None, random splitting is used
    train_ratio : float, optional
        Ratio of data for training when random splitting (default: 0.7)
    valid_ratio : float, optional
        Ratio of data for validation when random splitting (default: 0.2)
    test_ratio : float, optional
        Ratio of data for testing when random splitting (default: 0.1)
    
    Returns
    -------
    tuple
        (train_adata, valid_adata, test_adata)
    '''
    
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    
    shuffled = shuffle_adata(adata)
    
    
    control_index = adata.obs[adata.obs['dose'] == 0.0].index.tolist()
    
    if split_key is not None and split_key in adata.obs.columns:
        
        train_index = adata.obs[adata.obs[split_key] == 'train'].index.tolist()
        valid_index = adata.obs[adata.obs[split_key] == 'valid'].index.tolist()
        test_index = adata.obs[adata.obs[split_key] == 'test'].index.tolist()
    else:
        
        n_samples = len(adata)
        indices = np.arange(n_samples)
        np.random.seed(2025)  
        np.random.shuffle(indices)
        
        
        train_end = int(n_samples * train_ratio)
        valid_end = train_end + int(n_samples * valid_ratio)
        
        
        train_index = indices[:train_end].tolist()
        valid_index = indices[train_end:valid_end].tolist()
        test_index = indices[valid_end:].tolist()
    
    train_index = list(set(train_index + control_index))
    valid_index = list(set(valid_index + control_index))
    test_index = list(set(test_index + control_index))
    
    train_adata = shuffled[train_index, :] if len(train_index) > 0 else None
    valid_adata = shuffled[valid_index, :] if len(valid_index) > 0 else None
    test_adata = shuffled[test_index, :] if len(test_index) > 0 else None
    
    return train_adata, valid_adata, test_adata



def save_model(
    args,
    epoch,
    model,
    model_without_ddp,
    optimizer,
    loss_scaler,
    model_ema=None,
    optimizer_disc=None,
    save_ckpt_freq=1,
):
    output_dir = Path(args.output_dir)

    # if epoch != "best":
    #     epoch = epoch + 1
    epoch_name = str(epoch)

    if not getattr(args, "enable_deepspeed", False):
        checkpoint_paths = [output_dir / "checkpoint.pth"]
        if epoch == "best":
            checkpoint_paths = [
                output_dir / ("checkpoint-%s.pth" % epoch_name),
            ]
        elif (epoch) % save_ckpt_freq == 0:
            checkpoint_paths.append(output_dir / ("checkpoint-%s.pth" % epoch_name))

        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                # 'scaler': loss_scaler.state_dict(),
                "args": args,
            }
            if loss_scaler is not None:
                to_save["scaler"] = loss_scaler.state_dict()

            if model_ema is not None:
                to_save["model_ema"] = get_state_dict(model_ema)

            if optimizer_disc is not None:
                to_save["optimizer_disc"] = optimizer_disc.state_dict()

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        if model_ema is not None:
            client_state["model_ema"] = get_state_dict(model_ema)
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % epoch_name,
            client_state=client_state,
        )

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
        layer_names=None,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(
    parameters, norm_type: float = 2.0, layer_names=None
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device

    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        )
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))

        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(
                    f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}"
                )

    return total_norm


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
    warmup_steps=-1,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule
