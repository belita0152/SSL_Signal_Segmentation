import os
import glob
import pandas as pd

# Path
base_path = os.path.join(os.getcwd(), '..', 'shhs2_o')  # 현재 경로에 따라 path 수정
data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535

#######################################################################

# Parsing : segment 만들기
from parser import Segment
parser = Segment()
x1, y1 = parser.parser(data_path, mask_path)
print(x1.shape)  # (243207, 3000, 4)
print(y1.shape)  # (243207, 3000, 1)

# # 전체 x: (243207, 3000, 4), y: (243207, 3000, 1)
dataset = TorchDataset(transform=None)
train_data, tuning_data, eval_data = random_split(dataset,
                                                  [dataset.split['SSL'],
                                                   dataset.split['Tuning'],
                                                   dataset.split['Eval']])  # ratio or length
x_train = train_data.dataset[train_data.indices] # 현재 Tuple. [0] : x, [1] : y

print("Train: ", len(train_data))  # 170245
print("Tuning: ", len(tuning_data))  # 48642
print("Eval: ", len(eval_data))  # 24320

print(train_dataloader)
print(tuning_dataloader)
print(eval_dataloader)

#######################################################################

# Augmentation : 4가지 기법 중 랜덤하게 2가지 선택
augment = SigAugmentation()
from dataloader import *
x0 = train_data  # (170245, 3000, 4) - from dataloader

# [1] Random crop
x1 = augment.random_crop(x0)
print(x1.shape)

# [2] Gaussian Noise (Jitter) : 원본 신호에 노이즈 추가
x2 = augment.random_gaussian_noise(x0)
print(x2.shape)

# [3] Random Permutation
x3 = augment.random_permutation(x1)
print(x3.shape)

# [4] Temporal Cutout: 일부 cut -> 자른 부분은 평균으로 대체
x4 = augment.random_temporal_cutout(x1)
print(x4.shape)

#######################################################################

# Encoder -> DataLoader, Encoder block별 shape 확인
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

import pickle
import copy
with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

train_data = copy.deepcopy(train_data)

train_data = TupleDataset(train_data)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=64, drop_last=True)

train_features, train_labels = next(iter(train_dataloader))
print(len(train_data))        # 170245
print(len(train_dataloader))  # 2660
print(f"Feature batch shape: {train_features.size()}")  # torch.Size([64, 4, 3000])
print(f"Labels batch shape: {train_labels.size()}")  # torch.Size([64, 1, 3000])

model = VGG(in_channels=4, num_classes=6)
model.to(device)

train_features = train_features.to(device)
model_output = model(train_features)
print(model_output)
print(model_output.shape)


#########################################################################

# Linear Probing
import warnings
import numpy as np
import torch
import random
import argparse
import glob
import os
import sys

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader import TorchDataset
from torch.utils.data import DataLoader
from encoder import VGG
import torch.nn as nn
from torch import optim
from loss.simclr import SimCLR
from augmentation import SigAugmentation as SigAug

# from sklearn.decomposition import PCA
from torch_pca import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm

warnings.filterwarnings("ignore")

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
ckpt_dir = './checkpoint'
log_dir = './log'

def get_args():
    parser = argparse.ArgumentParser(description='SSL_Signal_Segmentation')
    # Dataset
    parser.add_argument('--sfreq',  default=10, type=int)
    parser.add_argument('--split_ratio', default={'Train': 0.68, 'Val': 0.02}, type=float)

    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=200, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--save_checkpoint', default=True, type=bool)
    parser.add_argument('--n_channels', default=4, type=int)
    parser.add_argument('--n_classes', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-8, type=float)
    parser.add_argument('--momentum', default=0.999, type=float)
    parser.add_argument('--device', default=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))

    return parser.parse_args()


def get_classification_accuracy(y, y_true):
    y_pred = np.zeros(y.shape[0], dtype=int)

    for i in range(y.shape[0]):
        mask_samples = y[:, i]
        if mask_samples.sum() == 0:
            y_pred[i] = 0
        else:
            mask = y[:, i] != 0
            y_pred[i] = np.unique(y[:, i][mask])

    return accuracy_score(y_pred, y_true)


# 1. Create dataset
args = get_args()
base_path = os.path.join(os.getcwd(), '..', 'shhs2_o')
data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535

# 2. Split into train / tuning / evaluation partitions
train_size = int(len(data_path) * args.split_ratio['Train'])
train_data_path =  data_path[0:train_size]
train_mask_path = mask_path[0:train_size]

val_size = int(len(data_path) * args.split_ratio['Val'])
val_data_path = data_path[train_size:(train_size+val_size)]
val_mask_path = mask_path[train_size:(train_size+val_size)]

train_data = TorchDataset(data_dir=train_data_path, mask_dir=train_mask_path, transform=None)
val_data = TorchDataset(data_dir=val_data_path, mask_dir=val_mask_path, transform=None)

# 3. Create dataloaders
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)

# 4. Set up
backbone = VGG(in_channels=args.n_channels, num_classes=args.n_classes)
backbone.to(args.device)

criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(backbone.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=len(train_dataloader),
                                                       eta_min=0,
                                                       last_epoch=-1)

# 5. Pre-train with SimCLR (Augmentation -> NTXentLoss)
sig_aug = SigAug()
simclr = SimCLR(input_dim=backbone.final_length, hidden_dim=64).to(args.device)

num_batch = len(train_dataloader)
for epoch in range(args.train_epochs):
    backbone.train()
    simclr.train()
    # (1) Train
    train_count = 0
    epoch_train_loss = []
    for batch, labels in train_dataloader:
        optimizer.zero_grad()
        x = batch.to(args.device)

        x1, x2 = sig_aug.convert_augmentation(x)

        h1, h2 = backbone(x1), backbone(x2)

        loss = simclr(h1, h2)
        loss.backward()
        optimizer.step()
        train_count += 1
        epoch_train_loss.append(float(loss.detach().cpu().item()))  # cpu에서 print

    epoch_train_loss = np.mean(np.array(epoch_train_loss))


    if (epoch + 1) % 1 == 0:
        # (2) Validation (with PCA) : https://www.datacamp.com/tutorial/principal-component-analysis-in-python
        backbone.eval()
        model = KNeighborsClassifier(n_neighbors=args.n_classes)

        with torch.no_grad():
            epoch_x, epoch_y = [], []
            for data in val_dataloader:
                x, y = data
                x, y = x.to(args.device), y.to(args.device)

                # 위에서 학습시킨 backbone model에 x를 통과시켜서 feature 추출
                x = backbone(x)  # x, y : torch.Size([1024, 256, 46]) -> 11776 // torch.Size([1024, 1, 3000]
                x = x.view(x.shape[0], -1)  # [1024, 11776]
                epoch_x.append(x)
                epoch_y.append(y)

        # EEG segmentation -> time point별 feature & label 필요
        epoch_x, epoch_y = torch.cat(epoch_x, dim=0), torch.cat(epoch_y, dim=0)  # cat. stack // concatenate. extend
        print(epoch_x.shape, epoch_y.shape)  # torch.Size([23552, 11776]) torch.Size([23552, 1, 3000])

        # x : PCA -> 차원축소
        pca = PCA(n_components=6)  # torch_pca
        epoch_x = pca.fit_transform(epoch_x)  # input : (n_samples, n_features)  (23552, 5)

        # kNN : https://github.com/eddymina/ECG_Classification_Pytorch/blob/master/ECG_notebook.ipynb
        epoch_x = epoch_x.detach().cpu().numpy()  # (23552, 5)
        epoch_y = epoch_y.view(-1, epoch_y.shape[-1]).detach().cpu().numpy()  # (23552, 3000)
        model.fit(epoch_x, epoch_y)
        epoch_pred = model.predict(epoch_x)
        print(epoch_pred.shape)

        val_acc = get_classification_accuracy(epoch_pred, epoch_y)
        val_mf1 = f1_score(y_true=epoch_y, y_pred=epoch_pred, average='macro')

        print('[Epoch] : {0:01d} \t '
              '[Train Loss] : {1:.4f} \t '
              '[Accuracy] : {2:2.4f}  \t '
              '[Macro-F1] : {3:2.4f}'.format(
            epoch + 1, epoch_train_loss, val_acc * 100, val_mf1 * 100))

    else:
        print('[Epoch] : {0:01d} \t '
              '[Train Loss] : {1:.4f} \t '.format(
            epoch + 1, epoch_train_loss))
