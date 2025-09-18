import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from matplotlib import rcParams
from torch.utils.tensorboard import SummaryWriter

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.acc        = []
        self.losses     = []
        self.val_loss   = []
        self.macro_precision = []
        self.macro_recall = []
        self.macro_f1 = []
        self.weighted_precision = []
        self.weighted_recall = []
        self.weighted_f1 = []

        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        device = next(model.parameters()).device
        dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1], device=device)
        self.writer.add_graph(model, dummy_input)

    def append_loss(self, epoch, acc, loss, val_loss,
                    macro_precision, macro_recall, macro_f1,
                    weighted_precision, weighted_recall, weighted_f1):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.acc.append(acc)
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        self.macro_precision.append(macro_precision)
        self.macro_recall.append(macro_recall)
        self.macro_f1.append(macro_f1)
        self.weighted_precision.append(weighted_precision)
        self.weighted_recall.append(weighted_recall)
        self.weighted_f1.append(weighted_f1)

        with open(os.path.join(self.log_dir, "epoch_acc.txt"), 'a') as f:
            f.write(str(acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_macro_precision.txt"), 'a') as f:
            f.write(str(macro_precision))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_macro_recall.txt"), 'a') as f:
            f.write(str(macro_recall))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_macro_f1.txt"), 'a') as f:  
            f.write(str(macro_f1))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_weighted_precision.txt"), 'a') as f:
            f.write(str(weighted_precision))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_weighted_recall.txt"), 'a') as f:
            f.write(str(weighted_recall))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_weighted_f1.txt"), 'a') as f:
            f.write(str(weighted_f1))
            f.write("\n")


        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        rcParams['font.family'] = 'Times New Roman'
        rcParams['axes.linewidth'] = 1.2   
        rcParams['axes.edgecolor'] = 'black'
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.direction'] = 'in'
        rcParams['xtick.labelsize'] = 14
        rcParams['ytick.labelsize'] = 14

        iters = range(len(self.losses))


        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(iters, self.losses, color='#D32F2F', linewidth=2.0, label='Train Loss')
        plt.plot(iters, self.val_loss, color='#1976D2', linewidth=2.0, label='Validation Loss')

        try:
            num = 5 if len(self.losses) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3),
                    color='#4CAF50', linestyle='--', linewidth=2.0, label='Smoothed Train Loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3),
                    color='#8E44AD', linestyle='--', linewidth=2.0, label='Smoothed Validation Loss')
        except Exception as e:
            # print(f"Smooth failed: {e}")
            pass

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"), dpi=300)
        plt.cla()
        plt.close("all")

        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(iters, self.acc, color='#D32F2F', linewidth=2.0, label='Accuracy')

        try:
            num = 5 if len(self.acc) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.acc, num, 3),
                    color='#4CAF50', linestyle='--', linewidth=2.0, label='Smoothed Accuracy')
        except Exception as e:
            # print(f"Smooth failed: {e}")
            pass

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"), dpi=300)
        plt.cla()
        plt.close("all")

        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(iters, self.macro_precision, color='#D32F2F', linewidth=2.0, label='Macro Precision')

        try:
            num = 5 if len(self.macro_precision) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.macro_precision, num, 3),
                 color='#4CAF50', linestyle='--', linewidth=2.0, label='Smoothed Macro Precision')
        except Exception as e:
            pass


        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Macro Precision', fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "epoch_macro_precision.png"), dpi=300)
        plt.cla()
        plt.close("all")


        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(iters, self.macro_recall, color='#D32F2F', linewidth=2.0, label='Macro Recall')


        try:
            num = 5 if len(self.macro_recall) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.macro_recall, num, 3),
                 color='#4CAF50', linestyle='--', linewidth=2.0, label='Smoothed Macro Recall')
        except Exception as e:
            pass


        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Macro Recall', fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "epoch_macro_recall.png"), dpi=300)
        plt.cla()
        plt.close("all")

        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(iters, self.macro_f1, color='#D32F2F', linewidth=2.0, label='Macro F1')


        try:
            num = 5 if len(self.macro_f1) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.macro_f1, num, 3),
                 color='#4CAF50', linestyle='--', linewidth=2.0, label='Smoothed Macro F1')
        except Exception as e:
            pass

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Macro F1', fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "epoch_macro_f1.png"), dpi=300)
        plt.cla()
        plt.close("all")

        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(iters, self.weighted_precision, color='#D32F2F', linewidth=2.0, label='Weighted Precision')

        try:
            num = 5 if len(self.weighted_precision) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.weighted_precision, num, 3),
                     color='#4CAF50', linestyle='--', linewidth=2.0, label='Smoothed Weighted Precision')
        except Exception:
            pass

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Weighted Precision', fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "epoch_weighted_precision.png"), dpi=300)
        plt.cla()
        plt.close("all")


        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(iters, self.weighted_recall, color='#D32F2F', linewidth=2.0, label='Weighted Recall')

        try:
            num = 5 if len(self.weighted_recall) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.weighted_recall, num, 3),
                     color='#4CAF50', linestyle='--', linewidth=2.0, label='Smoothed Weighted Recall')
        except Exception:
            pass

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Weighted Recall', fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "epoch_weighted_recall.png"), dpi=300)
        plt.cla()
        plt.close("all")


        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(iters, self.weighted_f1, color='#D32F2F', linewidth=2.0, label='Weighted F1')

        try:
            num = 5 if len(self.weighted_f1) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.weighted_f1, num, 3),
                     color='#4CAF50', linestyle='--', linewidth=2.0, label='Smoothed Weighted F1')
        except Exception:
            pass

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Weighted F1', fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "epoch_weighted_f1.png"), dpi=300)
        plt.cla()
        plt.close("all")