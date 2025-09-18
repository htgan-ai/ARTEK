import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.arcface import Arcface
from nets.arcface_training import get_lr_scheduler, set_optimizer_lr
from utils.callback import LossHistory
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate
from utils.utils import (get_num_classes, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch
from datetime import datetime
from muon import SingleDeviceMuonWithAuxAdam as MuonWithAuxAdam
import json
from pprint import pformat
from utils.losses import build_criterion
from collections import Counter
from torch.utils.data import WeightedRandomSampler


def _line_to_cid(ln: str) -> int:
    return int(ln.split(";", 1)[0])

def build_muon_param_groups(model,
                            conv_only: bool = True,
                            exclude_first_conv: bool = True,
                            muon_lr: float = 0.02,
                            adamw_lr: float = 3e-4,
                            wd: float = 0.01,
                            betas=(0.9, 0.95)):
    """
    Split the Arcface model into two groups according to Muon paper recommendations:
    - hidden_weights: Using Muon (default only convolution kernels; can switch to all weight matrices)
    - Others (bias/BN/classification head, etc.): Using AdamW
    """
    body = getattr(model, "arcface", None)   # backbone
    head = getattr(model, "head", None)     

    hidden_weights = []
    hidden_gains_biases = []
    nonhidden_params = []

    first_conv_w = None
    if body is not None:
        for m in body.modules():
            if isinstance(m, nn.Conv2d):
                first_conv_w = m.weight
                break

        seen = set()
        def add_once(p, bucket):
            if (p is None) or (not p.requires_grad):
                return
            pid = id(p)
            if pid in seen: return
            seen.add(pid)
            bucket.append(p)

        for name, p in body.named_parameters():
            if not p.requires_grad:
                continue
            if conv_only:

                if p.ndim >= 4:
                    if exclude_first_conv and (first_conv_w is p):
                        add_once(p, nonhidden_params)      
                    else:
                        add_once(p, hidden_weights)         
                else:
                    add_once(p, hidden_gains_biases)        
            else:

                if p.ndim >= 2:
                    if exclude_first_conv and (first_conv_w is p) and p.ndim >= 4:
                        add_once(p, nonhidden_params)
                    else:
                        add_once(p, hidden_weights)
                else:
                    add_once(p, hidden_gains_biases)


    if head is not None:
        for p in head.parameters():
            if p.requires_grad:
                nonhidden_params.append(p)

    param_groups = []
    if hidden_weights:
        param_groups.append(dict(params=hidden_weights, use_muon=True,
                                 lr=muon_lr, weight_decay=wd))
    if hidden_gains_biases or nonhidden_params:
        param_groups.append(dict(params=(hidden_gains_biases + nonhidden_params), use_muon=False,
                                 lr=adamw_lr, betas=betas, weight_decay=wd))
    return param_groups


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def dump_run_config(save_dir: str, cfg: dict, txt_name: str = "config.txt", json_name: str = "config.json"):
    _ensure_dir(save_dir)
    with open(os.path.join(save_dir, json_name), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)

def _read_list(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _collect_class_ids(lines):
    ids = set()
    for ln in lines:
        try:
            cid, _ = ln.split(";", 1)
            ids.add(int(cid))
        except Exception:
            pass
    return ids

def _class_counts(lines):
    cnt = Counter()
    for ln in lines:
        try:
            cid, _ = ln.split(';', 1)
            cnt[int(cid)] += 1
        except:
            pass
    max_c = max(cnt.keys())
    vec = [0]*(max_c+1)
    for k,v in cnt.items(): vec[k]=v
    return vec

if __name__ == "__main__":

    Cuda            = True

    seed            = 3047

    distributed     = False

    sync_bn         = False

    fp16            = False

    K_FOLD_DIR   = "splits_kfold/k5_seed42_aug/fold_5"
    TRAIN_LIST_P = os.path.join(K_FOLD_DIR, "train.txt")
    VAL_LIST_P   = os.path.join(K_FOLD_DIR, "val.txt")
    #--------------------------------------------------------#
    #--------------------------------------------------------#
    input_shape     = [256, 256, 3]
    #--------------------------------------------------------#
    #   artifingerNet   OUR
    #   mobilefacenet
    #   mobilenetv1
    #   iresnet18
    #   iresnet34
    #   iresnet50
    #   iresnet64
    #   iresnet100
    #   iresnet200
    #   edgeface_xs_gamma_06
    #   parameternet_600m
    #   ghostnetv3
    #   faster_vit_0_any_res  0-6
    #   shvit_s1 1-4
    #   efficientvim_m1 1-4
    #   groupmamba_tiny, groupmamba_small, groupmamba_base
    #   tinyvim_s, tinyvim_b, tinyvim_l

    # backbone        = "mobilefacenet"
    backbone        = "artifingerNet"
    embedding_size = 128
    head_type       = "cosface"   # "adaface", "arcface", "cosface", "curricularface"
    head_kwargs = {
        "m": 0.3,       # margin
        "s": 32,        
        "t_alpha": 0.8, 
        "h": 0.4,     
    }
    loss_name = 'nll'  # 'ce' 'nll' 
    model_path      = ""

    pretrained      = False

    Init_Epoch      = 0
    Epoch           = 50
    batch_size      = 80

    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.001

    optimizer_type      = "muon"
    momentum            = 0.9
    weight_decay        = 5e-4

    lr_decay_type       = "cos"

    save_period         = 99999999

    time_stamp = datetime.now().strftime("%Y_%m_%d_%H:%M")
    save_dir = f"logs/{K_FOLD_DIR}/{backbone}_{head_type}_{optimizer_type}_{loss_name}_{time_stamp}"

    num_workers     = 12

    lfw_eval_flag   = False

    lfw_dir_path    = "0524sft512ok"
    lfw_pairs_path  = "model_data/0524sft512ok_pair.txt"

    seed_everything(seed)

    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0


    train_lines = _read_list(TRAIN_LIST_P)
    val_lines   = _read_list(VAL_LIST_P)
    _class_ids = _collect_class_ids(train_lines) | _collect_class_ids(val_lines)
    num_classes = len(_class_ids)
    class_counts = _class_counts(train_lines)

    device = torch.device('cuda', local_rank) if Cuda else torch.device('cpu')
    model = Arcface(num_classes=num_classes, backbone=backbone, head_type=head_type, 
                    pretrained=pretrained, head_kwargs=head_kwargs, embedding_size=embedding_size)
    
    w=None

    criterion = build_criterion(name=loss_name, num_classes=num_classes,
                                class_counts=class_counts, class_weight=w).to(device)
    if model_path != '':

        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        

        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:

            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #----------------------#
    #   Loss
    #----------------------#
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None
    #---------------------------------#
    #   LFW
    #---------------------------------#
    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=32, shuffle=False) if lfw_eval_flag else None


    num_train = len(train_lines)
    num_val   = len(val_lines)

    show_config(
        num_classes = num_classes, backbone = backbone, head_type = head_type, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    if True:

        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        if optimizer_type == "muon":

            muon_cfg = dict(
                conv_only=True,             
                exclude_first_conv=True,    
                muon_lr=0.02,
                adamw_lr=3e-4,
                wd=0.01,
                betas=(0.9, 0.95),
            )
            param_groups = build_muon_param_groups(model, **muon_cfg)
            optimizer = MuonWithAuxAdam(param_groups)
            print("[Muon] groups:", len(optimizer.param_groups))
            for i, g in enumerate(optimizer.param_groups):
                print(f"  group{i}: use_muon={g.get('use_muon', False)}, "
                    f"wd={g.get('weight_decay')}, params={len(g['params'])}")

            for g in optimizer.param_groups:
                g.setdefault("base_lr", g["lr"])
        else:
            optimizer = {
                'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=0.0),
                'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=0.01),
                'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
            }[optimizer_type]
            for g in optimizer.param_groups:
                g.setdefault("base_lr", g["lr"])

        #---------------------------------------#
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        #---------------------------------------#
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

        #---------------------------------------#

        # train_dataset   = FacenetDataset(input_shape, lines[:num_train], random = True)
        # val_dataset     = FacenetDataset(input_shape, lines[num_train:], random = False)
        train_dataset   = FacenetDataset(input_shape, train_lines, random=True, 
                                         k90_prob=0.5, small_rotate_prob=0.5, small_rotate_max_deg=15)
        val_dataset     = FacenetDataset(input_shape, val_lines,   random=False)

        INTEREST_MIN, INTEREST_MAX = 0, 10
        W_HI, W_LO = 1.0, 0.1   

        train_labels = [_line_to_cid(ln) for ln in train_lines]
        weights_per_sample = [
            (W_HI if (INTEREST_MIN <= y <= INTEREST_MAX) else W_LO)
            for y in train_labels
        ]

        weighted_sampler = WeightedRandomSampler(
            weights=weights_per_sample,
            num_samples=len(weights_per_sample),
            replacement=True
        )
        # ===== END: Weighted sampler =====
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            # train_sampler   = weighted_sampler
            train_sampler   = None
            val_sampler     = None
            shuffle         = True 

        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=train_sampler, 
                                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=val_sampler, 
                                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        
        best_val_acc = 0.0
        best_lfw_acc = 0.0
        best_epoch_metrics_all_time = None

     
        global_cfg = {
            "time_stamp": time_stamp,
            "save_dir": save_dir,

            
            "Cuda": Cuda,
            "distributed": distributed,
            "sync_bn": sync_bn,
            "fp16": fp16,
            "seed": seed,
            "ngpus_per_node": ngpus_per_node,

           
            "K_FOLD_DIR": K_FOLD_DIR,
            "TRAIN_LIST_P": TRAIN_LIST_P,
            "VAL_LIST_P": VAL_LIST_P,
            "num_train": num_train,
            "num_val": num_val,
            "num_classes": num_classes,
            "input_shape": input_shape,

            
            "backbone": backbone,
            "embedding_size": embedding_size,
            "head_type": head_type,
            "head_kwargs": head_kwargs,     
            "model_path": model_path,
            "pretrained": pretrained,

            
            "criterion": loss_name,
            "Init_Epoch": Init_Epoch,
            "Epoch": Epoch,
            "batch_size": batch_size,
            "optimizer_type": optimizer_type,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "lr_decay_type": lr_decay_type,
            "Init_lr": Init_lr,
            "Min_lr": Min_lr,
            "Init_lr_fit": Init_lr_fit,
            "Min_lr_fit": Min_lr_fit,
            "save_period": save_period,
            "num_workers": num_workers,

            
            "lfw_eval_flag": lfw_eval_flag,
            "lfw_dir_path": lfw_dir_path,
            "lfw_pairs_path": lfw_pairs_path,
        }

        if optimizer_type == "muon":

            global_cfg["muon_cfg"] = muon_cfg
            pg_summ = []
            for i, g in enumerate(optimizer.param_groups):
                pg_summ.append({
                    "group_idx": i,
                    "use_muon": g.get("use_muon", False),
                    "weight_decay": g.get("weight_decay"),
                    "base_lr": g.get("base_lr", g.get("lr")),
                    "num_params": len(g["params"]),
                })
            global_cfg["param_groups_summary"] = pg_summ

        if (not distributed) or (distributed and local_rank == 0):
            dump_run_config(save_dir, global_cfg)
        
        
        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            best_val_acc, best_lfw_acc, current_metrics, best_metrics = fit_one_epoch(
                model_train, model, loss_history, optimizer,
                epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, Cuda,
                LFW_loader, lfw_eval_flag, fp16, scaler, save_period, save_dir,
                local_rank=local_rank,
                best_val_acc=best_val_acc, best_lfw_acc=best_lfw_acc,
                criterion=criterion,
            )
            if best_metrics is not None:
                best_epoch_metrics_all_time = best_metrics


        if local_rank == 0:
            loss_history.writer.close()
            print("\n================ Final Metrics ================")
            if best_epoch_metrics_all_time is not None:
                bm = best_epoch_metrics_all_time
                print(
                    f"Best@Epoch {bm['epoch']}: "
                    f"val_acc={bm['val_acc']:.6f}, val_loss={bm['val_loss']:.6f}\n"
                    f"Macro  -> P={bm['macro_precision']:.6f}, R={bm['macro_recall']:.6f}, F1={bm['macro_f1']:.6f}\n"
                    f"Weighted -> P={bm['weighted_precision']:.6f}, R={bm['weighted_recall']:.6f}, F1={bm['weighted_f1']:.6f}"
                )
                results_dict = {
                    "best_val_acc": best_val_acc,
                    "best_epoch_metrics": best_epoch_metrics_all_time,
                }
                with open(os.path.join(save_dir, "results_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(results_dict, f, ensure_ascii=False, indent=2, default=str)
            else:
                print("No best epoch recorded (check training run).")
            print("==============================================\n")
