import warnings
import numpy as np
import torch
import random
import argparse
import glob
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader import TorchDataset
from torch.utils.data import DataLoader, random_split
from encoder import VGG
import torch.nn as nn
from torch import optim
from loss.simclr import SimCLR
from augmentation import SigAugmentation as SigAug

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings("ignore")

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt_dir = './checkpoint'
log_dir = './log'

def get_args():
    parser = argparse.ArgumentParser(description='SSL_Signal_Segmentation')
    # Dataset
    parser.add_argument('--sfreq',  default=10, type=int)
    parser.add_argument('--split_ratio', default={'Train': 0.6, 'Eval': 0.1}, type=float)

    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=200, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--save_checkpoint', default=True, type=bool)
    parser.add_argument('--n_channels', default=4, type=int)
    parser.add_argument('--n_classes', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-8, type=float)
    parser.add_argument('--momentum', default=0.999, type=float)
    parser.add_argument('--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    return parser.parse_args()


# 1. Create dataset
args = get_args()
base_path = os.path.join(os.getcwd(), '..', 'shhs2_o')
data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535
dataset = TorchDataset(data_dir=data_path, mask_dir=mask_path, transform=None)

# 2. Split into train / tuning / evaluation partitions
dataset_size = len(dataset)
train_size = int(dataset_size * args.split_ratio['Train'])
eval_size = dataset_size - train_size

train_data, eval_data = random_split(dataset, [train_size, eval_size])

# 3. Create dataloaders
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

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


    if (epoch + 1) % 2 == 0:
        # (2) Evaluate
        epoch_x, epoch_y = [], []
        model = KNeighborsClassifier(n_neighbors=args.n_classes)

        from tqdm import tqdm
        for data in tqdm(eval_dataloader):
            backbone.eval()
            with torch.no_grad():
                x, y = data
                x, y = x.to(device), y.to(device)  # cpu -> gpu에서 tensor 계산
                x = backbone(x)  # 위에서 학습시킨 backbone model에 x를 통과시켜서 feature 추출

                b = x.shape[0]  # batch 크기
                x, y = x.view(b, -1), y.view(b, -1)  # 2차원으로 변환
                epoch_x.append(x.detach().cpu().numpy())
                epoch_y.append(y.detach().cpu().numpy())

        epoch_x, epoch_y = np.concatenate(epoch_x, axis=0), np.concatenate(epoch_y, axis=0)  # .extend 사용 가능
    
        pca = PCA(n_components=50)
        epoch_x = pca.fit_transform(epoch_x)

        model.fit(epoch_x, epoch_y)
        epoch_pred = model.predict(epoch_x)

        eval_acc = accuracy_score(y_true=epoch_y, y_pred=epoch_pred)
        eval_mf1 = f1_score(y_true=epoch_y, y_pred=epoch_pred, average='macro')

        print('[Epoch] : {0:01d} \t '
              '[Train Loss] : {1:.4f} \t '
              '[Accuracy] : {1:2.4f}  \t '
              '[Macro-F1] : {2:2.4f}'.format(
            epoch + 1, epoch_train_loss, eval_acc * 100, eval_mf1 * 100))

    else:
        print('[Epoch] : {0:01d} \t '
              '[Train Loss] : {1:.4f} \t '.format(
            epoch + 1, epoch_train_loss))
