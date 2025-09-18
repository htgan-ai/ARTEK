import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .utils import get_lr
from .utils_metrics import evaluate


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, 
                  epoch_step_val, gen, gen_val, Epoch, cuda, test_loader, lfw_eval_flag, 
                  fp16, scaler, save_period, save_dir, local_rank=0,
                  best_val_acc=0.0, best_lfw_acc=0.0, criterion=None):
    total_loss          = 0
    total_accuracy      = 0

    val_total_loss      = 0
    val_total_accuracy  = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    # if criterion is None:
    #     criterion = nn.NLLLoss()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                labels  = labels.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs     = model_train(images, labels, mode="train")
            # loss        = nn.NLLLoss()(F.log_softmax(outputs, -1), labels)
            loss        = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs     = model_train(images, labels, mode="train")
                # loss        = nn.NLLLoss()(F.log_softmax(outputs, -1), labels)
                loss        = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            train_acc         = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
            
        total_loss      += loss.item()
        total_accuracy  += train_acc.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'accuracy'  : total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    val_correct = torch.tensor(0, device=torch.device(f'cuda:{local_rank}') if cuda else 'cpu', dtype=torch.long)
    val_count   = torch.tensor(0, device=torch.device(f'cuda:{local_rank}') if cuda else 'cpu', dtype=torch.long)
    conf_mat = None
    num_classes = None
    
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, labels = batch

        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                labels  = labels.cuda(local_rank)

            optimizer.zero_grad()
            outputs     = model_train(images, labels, mode="train")
            # loss        = nn.NLLLoss()(F.log_softmax(outputs, -1), labels)
            loss        = criterion(outputs, labels)
            if num_classes is None:
                num_classes = outputs.size(-1)
                conf_mat = torch.zeros((num_classes, num_classes),
                                    device=labels.device, dtype=torch.long)
            preds = torch.argmax(outputs, dim=-1)
            val_correct += (preds == labels).sum()
            val_count   += labels.numel()
            k = (labels * num_classes + preds).to(torch.int64)
            cm = torch.bincount(k, minlength=num_classes * num_classes)
            conf_mat += cm.view(num_classes, num_classes)
            batch_acc = (preds == labels).float().mean().item()

            # val_acc    = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
            
            val_total_loss      += loss.item()
            # val_total_accuracy  += val_acc.item()
            val_total_accuracy  += batch_acc

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_total_loss / (iteration + 1),
                                'accuracy'  : val_total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    val_loss_epoch = val_total_loss / max(1, epoch_step_val)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count,   op=dist.ReduceOp.SUM)
        dist.all_reduce(conf_mat,    op=dist.ReduceOp.SUM)
    val_acc_epoch = (val_correct.float() / val_count.clamp(min=1).float()).item()
    conf = conf_mat.float()
    tp = torch.diag(conf)                               # (C,)
    support = conf.sum(dim=1)                          
    predicted = conf.sum(dim=0)                         
    precision_per_class = tp / predicted.clamp(min=1)
    recall_per_class    = tp / support.clamp(min=1)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class).clamp(min=1e-12)
    macro_precision = precision_per_class.mean().item()
    macro_recall    = recall_per_class.mean().item()
    macro_f1        = f1_per_class.mean().item()
    weights = support / support.sum().clamp(min=1)
    weighted_precision = (precision_per_class * weights).sum().item()
    weighted_recall    = (recall_per_class * weights).sum().item()
    weighted_f1        = (f1_per_class * weights).sum().item()
    current_metrics = {
        "epoch": epoch + 1,
        "val_loss": val_loss_epoch,
        "val_acc":  val_acc_epoch,
        "macro_precision":   macro_precision,
        "macro_recall":      macro_recall,
        "macro_f1":          macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall":    weighted_recall,
        "weighted_f1":        weighted_f1,
    }
    # val_acc_epoch  = val_total_accuracy / max(1, epoch_step_val)

    # ==================== LFW Eval (optional) ====================
    lfw_acc_mean = None
    if lfw_eval_flag and (test_loader is not None):
        if local_rank == 0:
            print("Start the verification of the LFW dataset")
        labels_list, dists_list = [], []
        for _, (data_a, data_p, label) in enumerate(test_loader):
            with torch.no_grad():
                data_a = data_a.float()
                data_p = data_p.float()
                if cuda:
                    data_a = data_a.cuda(local_rank, non_blocking=True)
                    data_p = data_p.cuda(local_rank, non_blocking=True)
                feat_a = model_train(data_a)
                feat_p = model_train(data_p)
                dists  = torch.sqrt(torch.sum((feat_a - feat_p) ** 2, dim=1))
            dists_list.append(dists.cpu().numpy())
            labels_list.append(label.cpu().numpy())

        labels    = np.concatenate(labels_list, axis=0)
        distances = np.concatenate(dists_list, axis=0)
        _, _, lfw_acc_cv, _, _, _, _ = evaluate(distances, labels)
        lfw_acc_mean = float(np.mean(lfw_acc_cv))
        lfw_acc_std  = float(np.std(lfw_acc_cv))
        if local_rank == 0:
            print(f'LFW_Accuracy: {lfw_acc_mean:.5f} +/- {lfw_acc_std:.5f}')

    

    best_metrics = None
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        print(
            f'[Epoch {epoch+1}] '
            f'train_loss={total_loss / epoch_step:.4f} | '
            f'val_loss={val_loss_epoch:.4f} | '
            f'val_acc={val_acc_epoch:.5f}'
        )
        print(
            f'Val Macro:  Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f} | '
            f'Weighted: Precision={weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1={weighted_f1:.4f}'
        )

        loss_history.append_loss(epoch, val_acc_epoch, total_loss / epoch_step, val_loss_epoch,
                                 macro_precision, macro_recall, macro_f1,
                                 weighted_precision, weighted_recall, weighted_f1
                                 )
        

        if val_acc_epoch > best_val_acc:
            old_best_path = os.path.join(save_dir, f'best_val_acc_{best_val_acc:.3f}.pth')
            if best_val_acc > 0 and os.path.exists(old_best_path):
                os.remove(old_best_path)
                print(f'Removed old best val acc file: {old_best_path}')
            best_val_acc = val_acc_epoch
            best_path = os.path.join(save_dir, f'best_val_acc_{val_acc_epoch:.3f}.pth')
            torch.save(model.state_dict(), best_path)
            print(f'New best val_acc: {best_val_acc:.5f}. Saved: {best_path}')
            best_metrics = dict(current_metrics)


        if lfw_acc_mean is not None and lfw_acc_mean > best_lfw_acc:
            old_best_lfw_path = os.path.join(save_dir, f'best_lfw_acc_{best_lfw_acc:.3f}.pth')
            if best_lfw_acc > 0 and os.path.exists(old_best_lfw_path):
                os.remove(old_best_lfw_path)
                print(f'Removed old best LFW acc file: {old_best_lfw_path}')
            best_lfw_acc = lfw_acc_mean
            best_lfw_path = os.path.join(save_dir, f'best_lfw_acc_{lfw_acc_mean:.3f}.pth')
            torch.save(model.state_dict(), best_lfw_path)
            print(f'New best LFW acc: {best_lfw_acc:.5f}. Saved: {best_lfw_path}')

        print('Total Loss: %.4f' % (total_loss / epoch_step))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch+1), total_loss / epoch_step, val_total_loss / epoch_step_val)))
    
    return best_val_acc, best_lfw_acc, current_metrics, best_metrics