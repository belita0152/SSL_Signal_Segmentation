import warnings
import numpy as np
import torch
import random
import argparse
import glob
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.tensorboard import SummaryWriter
from dataloader import TorchDataset
from torch.utils.data import DataLoader
from encoder import VGG
import torch.nn as nn
from torch import optim
from loss.simclr import SimCLR
from augmentation import SigAugmentation

# from sklearn.decomposition import PCA
from torch_pca import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
ckpt_dir = './checkpoint_VGG_enc_no_dropout'
log_dir = './log'

writer = SummaryWriter(log_dir=log_dir)

def get_args():
    parser = argparse.ArgumentParser(description='SSL_Signal_Segmentation')
    # Dataset
    parser.add_argument('--sfreq',  default=10, type=int)
    parser.add_argument('--split_ratio', default={'Train': 0.65, 'Val': 0.05})

    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=200, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--save_checkpoint', default=True, type=bool)
    parser.add_argument('--n_channels', default=4, type=int)
    parser.add_argument('--n_classes', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.999, type=float)
    parser.add_argument('--device', default=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))

    return parser.parse_args()


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
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

# 4. Set up
backbone = VGG(in_channels=args.n_channels)
backbone.to(args.device)

criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.Adam(backbone.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=len(train_dataloader),
                                                       eta_min=0,
                                                       last_epoch=-1)
scaler = torch.cuda.amp.GradScaler()

# 5. Pre-train with SimCLR (Augmentation -> NTXentLoss)
sig_aug = SigAugmentation()
simclr = SimCLR(input_dim=backbone.final_length, hidden_dim=64).to(args.device)

num_batch = len(train_dataloader)
best_f1, best_epoch = 0.0, 0

for epoch in range(args.train_epochs):
    # (1) Train
    backbone.train()
    simclr.train()

    train_count = 0
    epoch_train_loss = []
    for batch, labels in train_dataloader:
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            x = batch.to(args.device)

            x1, x2 = sig_aug.convert_augmentation(x)

            h1, h2 = backbone(x1), backbone(x2)

            loss = simclr(h1, h2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_count += 1
        epoch_train_loss.append(float(loss.detach().cpu().item()))  # cpu에서 print

    epoch_train_loss = np.mean(np.array(epoch_train_loss))

    if (epoch + 1) % 1 == 0:
        # (2) Validation (with PCA) : https://www.datacamp.com/tutorial/principal-component-analysis-in-python
        backbone.eval()
        model = KNeighborsClassifier(n_neighbors=2)
        epoch_x, epoch_y = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch
                x = x.to(args.device)
                y = y.to(args.device)

                x = backbone(x)
                x = x.permute(0, 2, 1)
                x = x.reshape(x.shape[0], -1)
                epoch_x.append(x)

                if y.dim() == 3:
                    y = y.squeeze(1)

                epoch_y.append(y)

        epoch_x, epoch_y = torch.cat(epoch_x, dim=0), torch.cat(epoch_y, dim=0).squeeze()

        pca = PCA(n_components=50)  # torch_pca // sklearn_pca
        epoch_x = pca.fit_transform(epoch_x)  # input : (n_samples, n_features)  (23552, 5)

        # kNN : https://github.com/eddymina/ECG_Classification_Pytorch/blob/master/ECG_notebook.ipynb
        epoch_x, epoch_y = epoch_x.detach().cpu().numpy(), epoch_y.detach().cpu().numpy()
        model.fit(epoch_x, epoch_y)
        epoch_pred = model.predict(epoch_x)

        val_acc = accuracy_score(epoch_y, epoch_pred)
        val_mf1 = f1_score(y_true=epoch_y, y_pred=epoch_pred, average='macro')
        conf_matrix = confusion_matrix(y_true=epoch_y, y_pred=epoch_pred)

        print('[Epoch] : {0:01d} \t '
              '[Train Loss] : {1:.4f} \t '
              '[Accuracy] : {2:2.4f}  \t '
              '[Macro-F1] : {3:2.4f}'.format(
            epoch + 1, epoch_train_loss, val_acc * 100, val_mf1 * 100))

        print(conf_matrix)

        writer.add_scalar('Loss/Epoch', epoch_train_loss, epoch + 1)
        writer.add_scalar('Val/Accuracy', val_acc * 100, epoch + 1)
        writer.add_scalar('Val/Macro-F1', val_mf1 * 100, epoch + 1)
        torch.save(backbone.state_dict(), f'{ckpt_dir}/VGG_encoder_epoch{epoch + 1}.pth')

        if val_mf1 > best_f1:
            best_f1 = val_mf1
            best_epoch = epoch + 1
            print(f'Best model: Epoch {best_epoch} with [Macro-F1] {best_f1:.4f}')

writer.close()
