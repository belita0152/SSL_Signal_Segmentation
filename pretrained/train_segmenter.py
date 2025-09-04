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
import torch.nn as nn
from torch import optim
from baseline.segmenter import Segmenter
from baseline.focal_loss import FocalLoss
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
ckpt_dir = './checkpoint_VGG_250902'
log_dir = './log_250902'

writer = SummaryWriter(log_dir=log_dir)

def get_args():
    parser = argparse.ArgumentParser(description='SSL_Signal_Segmentation')
    # Dataset
    parser.add_argument('--sfreq',  default=10, type=int)
    parser.add_argument('--split_ratio', default={'Train': 0.65, 'Val': 0.05})

    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=200, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--save_checkpoint', default=True, type=bool)
    parser.add_argument('--n_channels', default=4, type=int)
    parser.add_argument('--n_classes', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.999, type=float)
    parser.add_argument('--device', default=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))

    return parser.parse_args()


def get_focal_loss(x, y):
    x = x.transpose(0, 2, 1)
    x = x.reshape(-1, x.shape[2])
    y = y.reshape(-1)

    loss = FocalLoss().to(args.device)

    return loss(x, y)



if __name__ == '__main__':

    # 1. Create dataset
    args = get_args()
    base_path = os.path.join(os.getcwd(), '../../../../data/segmentation/shhs2_o')
    data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
    mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535

    # 2. Split into train / tuning / evaluation partitions
    train_size = int(len(data_path) * args.split_ratio['Train'])
    train_data_path =  data_path[0:train_size]
    train_mask_path = mask_path[0:train_size]

    val_size = int(len(data_path) * args.split_ratio['Val'])
    val_data_path = data_path[train_size:(train_size+val_size)]
    val_mask_path = mask_path[train_size:(train_size+val_size)]

    train_data = TorchDataset(data_dir=train_data_path, mask_dir=train_mask_path)
    val_data = TorchDataset(data_dir=val_data_path, mask_dir=val_mask_path)

    # 3. Create dataloaders
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 4. Set up
    model = Segmenter(n_cls=6)
    model.to(args.device)

    # get_loss = FocalLoss(task_type='multi-label', num_classes=args.n_classes)
    get_loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=len(train_dataloader),
                                                           eta_min=0,
                                                           last_epoch=-1)
    scaler = torch.cuda.amp.GradScaler()

    # 5. Pre-train with Segmenter
    num_batch = len(train_dataloader)
    best_f1, best_epoch = 0.0, 0
    best_model_state, best_score = model.state_dict(), 0

    for epoch in range(args.train_epochs):
        # (1) Train
        model.train()

        train_count = 0
        epoch_train_loss = []
        for batch, labels in train_dataloader:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                x = batch.to(args.device)
                labels = labels.long().to(args.device)
                output = model(x)  # output: [16, 6, 3000] / label: [16, 3000]

                loss = get_loss(output, labels)
                # loss = get_focal_loss(output, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_count += 1
            epoch_train_loss.append(float(loss.detach().cpu().item()))  # cpu에서 print

        epoch_train_loss = np.mean(np.array(epoch_train_loss))

        with torch.no_grad():
            preds, labels = [], []
            for batch, label in val_dataloader:
                model.eval()

                batch = batch.to(args.device)
                label = label.long().to(args.device)
                output = model(batch)  # [16, 6, 3000]
                output = torch.argmax(output, dim=1)  # [16, 3000]
                preds.append(output)
                labels.append(label)

            preds, labels = torch.cat(preds, dim=0), torch.cat(labels, dim=0).squeeze()
            acc = (preds == labels).float().mean().item()

            preds, labels = preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
            preds = preds.reshape(-1)
            labels = labels.reshape(-1)

            val_acc = balanced_accuracy_score(preds, labels)
            val_mf1 = f1_score(y_true=labels, y_pred=preds, average='macro')

            print('[Epoch] : {0:01d} \t '
                  '[Train Loss] : {1:.4f} \t '
                  '[Accuracy] : {2:2.4f}  \t '
                  '[Macro-F1] : {3:2.4f}'.format(
                epoch + 1, epoch_train_loss, val_acc * 100, val_mf1 * 100))

    writer.close()