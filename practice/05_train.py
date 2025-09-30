"""

Step 1. Encoder 학습시키기

"""

import warnings
import numpy as np
import torch
import random
import argparse
from dataloader import TorchDataset
from utils import *
from torch.utils.data import DataLoader, random_split, Dataset
# from torch.utils.tensorboard import SummaryWriter
from torch import optim
from encoder import TupleDataset, VGG
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss.simclr import SimCLR
from augmentation import SigAugmentation as SigAug


warnings.filterwarnings("ignore")

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt_dir = './checkpoint'
log_dir = './log'

def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                   './%s/model_epoch%d.pth' % (ckpt_dir, epoch))

import matplotlib.pyplot as plt
def plot_results(results):
    # x : epoch, y : train loss
    data = np.array(results)
    x, y = data[:, 0], data[:, 1]

    fig = plt.figure()
    fig.set_size_inches(8, 8)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=-70, elev=30)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Loss')
    ax.plot_trisurf(x, y, cmap='viridis')
    plt.savefig('./loss.png', dpi=100)
    plt.show()



def get_args():
    parser = argparse.ArgumentParser(description='SSL_Signal_Segmentation')
    # Dataset
    parser.add_argument('--sfreq',  default=100, type=int)
    parser.add_argument('--split_ratio', default={'SSL': 0.7, 'Tuning': 0.2, 'Eval': 0.1}, type=float)

    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=10, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_checkpoint', default=True, type=bool)
    parser.add_argument('--n_channels', default=4, type=int)
    parser.add_argument('--n_classes', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-8, type=float)
    parser.add_argument('--momentum', default=0.999, type=float)
    parser.add_argument('--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    return parser.parse_args()


# 1. Create dataset
args = get_args()
# dataset = TorchDataset(transform=None)  # class 안에서 파싱 함수 사용하였음
# # #
# # # 2. Split into train / tuning / evaluation partitions
# dataset_size = len(dataset)
# train_size = int(dataset_size * args.split_ratio['SSL'])
# tuning_size = int(dataset_size * args.split_ratio['Tuning'])
# eval_size = dataset_size - train_size - tuning_size
# train_data, tuning_data, eval_data = random_split(dataset,
#                                                   [train_size, tuning_size, eval_size])

import pickle
with open('train_dataloader.pkl', 'rb') as f:
    train_data = pickle.load(f)


# 3. Create data loaders
loader_args = dict(batch_size=args.batch_size, pin_memory=True)
# train_data = TupleDataset(train_data)  # Tensor Dataset 형태로 저장
train_dataloader = DataLoader(train_data, shuffle=True, **loader_args)


# 4. Set up the model, optimizer, loss, lr, and train function
backbone = VGG(in_channels=args.n_channels, num_classes=args.n_classes)
backbone.to(args.device)

import torch.nn as nn
criterion = nn.CrossEntropyLoss(reduction='sum')  # softmax + CrossEntropy
optimizer = optim.Adam(backbone.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # foreach -> don't know properly
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=0,
                                                       last_epoch=-1)

# 5. Pre-Train with SimCLR (Augmentation -> NTXentLoss)
# (Train Loop) Projection Head -> NTXentLoss -> Save checkpoints
sig_aug = SigAug()
simclr = SimCLR(input_dim=backbone.final_length, hidden_dim=64).to(args.device)
# tensorboard_path = os.path.join(log_dir, 'VGG', 'tensorboard')
# tensorboard_writer = SummaryWriter(log_dir=tensorboard_path)

num_batch = len(train_dataloader)  # 2661


for epoch in range(args.train_epochs):
    train_count = 0

    epoch_loss = []
    for batch, labels in train_dataloader:  # batch 64: torch.Size([64, 4, 3000])
        # Train => SimCLR Framework: https://amitness.com/posts/simclr
        optimizer.zero_grad()

        x = batch.to(args.device)
        # print(x.shape)
        x1, x2 = sig_aug.convert_augmentation(x)

        h1, h2 = backbone(x1), backbone(x2)  # Backbone encoder
        # print(h1.shape, h2.shape)  # torch.Size([64, 256, 1500]), torch.Size([64, 256, 1500])

        loss = simclr(h1, h2)
        loss.backward()
        optimizer.step()
        train_count += 1
        epoch_loss.append(float(loss.detach().cpu().item()))  # cpu에서 print
        # print(epoch_loss)

    epoch_loss = np.mean(np.array(epoch_loss))
    print('[Epoch] : {0:01d} \t '
          '[Train Loss] : {1:.4f} \t '.format(
        epoch + 1, epoch_loss))





    # # print(f"Epoch: {epoch}\tLoss: {np.mean(np.array(epoch_loss))}")
    # save(ckpt_dir=ckpt_dir, net=simclr, optim=optimizer, epoch=epoch)
