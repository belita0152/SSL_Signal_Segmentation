import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse

import pickle
import torch
from unet import UNet
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter
log_dir = './logdir'
writer = SummaryWriter(log_dir=log_dir)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

with open('tuning_x.pkl', 'rb') as f:
    tuning_x = pickle.load(f)

with open('tuning_y.pkl', 'rb') as f:
    tuning_y = pickle.load(f)

with open('eval_x.pkl', 'rb') as f:
    eval_x = pickle.load(f)

with open('eval_y.pkl', 'rb') as f:
    eval_y = pickle.load(f)


class TupleDataset(Dataset):  # 일종의 get data
    def __init__(self, x, y):
        self.features = x  # x data
        self.labels = y  # y labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_args():
    parser = argparse.ArgumentParser(description='Signal_Segmentation_Finetuning')
    # Dataset
    parser.add_argument('--sfreq',  default=10, type=int)
    parser.add_argument('--split_ratio', default={'SSL': 0.7, 'Tuning': 0.2, 'Eval': 0.1}, type=float)

    # Train Hyperparameter
    parser.add_argument('--finetuning_epochs', default=200, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--save_checkpoint', default=True, type=bool)
    parser.add_argument('--n_channels', default=4, type=int)
    parser.add_argument('--n_classes', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-8, type=float)
    parser.add_argument('--momentum', default=0.999, type=float)
    parser.add_argument('--device', default=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))

    return parser.parse_args()

def crop_target_to_match(y, target_length):
    total_length = y.shape[1]  # e.g., 3000
    start = (total_length - target_length) // 2
    end = start + target_length
    return y[:, start:end]


def get_loss(pred, real, w, smooth):
    # 1. CE Loss
    loss = criterion(pred, real)

    # 2. Dice Loss
    n_classes = pred.shape[1]
    pred = torch.softmax(pred, dim=1)

    real = torch.nn.functional.one_hot(real, num_classes=n_classes)  # [B, L, C]
    real = real.permute(0, 2, 1).float()  # [B, C, L]

    intersection = (pred * real).sum(dim=(0, 2))
    union = pred.sum(dim=(0, 2)) + real.sum(dim=(0, 2))

    dice = (2. * intersection + smooth) / (union + smooth)
    dice = dice.mean()
    dice_loss = 1 - dice

    # 3. Total Loss (weighted)
    total_loss = loss * w + dice_loss

    return loss, dice_loss, total_loss


def get_performance(pred, true, n_classes, smooth):
    # 1. dice_score
    n_classes = pred.shape[1]
    pred = torch.softmax(pred, dim=1)

    real = torch.nn.functional.one_hot(true, num_classes=n_classes)  # [B, L, C]
    real = real.permute(0, 2, 1).float()  # [B, C, L]

    def get_dice_score(pred, real, smooth):
        intersection = (pred * real).sum(dim=(0, 2))
        union = pred.sum(dim=(0, 2)) + real.sum(dim=(0, 2))

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_score = dice.mean()
        return dice_score.detach().cpu().numpy()

    # 2. mIoU
    def get_iou(pred, real, n_classes):
        iou_per_class = []
        pred = torch.argmax(pred, dim=1)

        for cls in range(n_classes):
            pred_c = pred
            # pred_c = pred[:, cls, :]  # [B, L]
            real_c = real[:, cls, :]  # [B, L]

            intersection = (pred_c * real_c).sum()
            union = pred_c.sum() + real_c.sum() - intersection

            if union == 0:
                iou_score = torch.tensor(float('nan'))
            else:
                iou_score = intersection / union

            iou_per_class.append(iou_score.detach().cpu().numpy())

        miou_score = np.nanmean(np.array(iou_per_class))

        return miou_score

    def get_pixel_accuracy(pred, real):
        pred = torch.argmax(pred, dim=1)
        correct = (pred == true).sum().item()
        total = true.numel()
        pixel_acc = correct / total

        return pixel_acc

    dice_score = get_dice_score(pred, real, smooth)
    miou = get_iou(pred, real, n_classes)
    pixel_acc = get_pixel_accuracy(pred, real)

    return dice_score, miou, pixel_acc


args = get_args()
tuning_dataset = TupleDataset(tuning_x, tuning_y)
tuning_dataloader = DataLoader(tuning_dataset, batch_size=args.batch_size, shuffle=True)
eval_dataset = TupleDataset(eval_x, eval_y)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)

enc_path = '/home/lhy/SSL_Signal_Segmentation/pretrained/checkpoint_VGG_enc/VGG_encoder_epoch82.pth'
model = UNet(num_classes=args.n_classes, encoder_ckpt=enc_path)  # class 안에서 freeze backbone
model = model.to(device)

criterion = nn.CrossEntropyLoss(reduction='mean')  # softmax + CrossEntropy
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # foreach -> don't know properly
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetuning_epochs)

for epoch in range(args.finetuning_epochs):
    model.train()  # unet enc(frozen)+dec 에 학습
    epoch_loss, epoch_dice_loss, epoch_total_loss = [], [], []
    epoch_dice_score, epoch_miou, epoch_pixel_acc = [], [], []
    for batch in tuning_dataloader:
        optimizer.zero_grad()
        x, y = batch
        x = x.permute(0, 2, 1)
        x, y = x.to(device), y.to(device)  # torch.Size([1024, 3000, 1])
        y = y.squeeze()

        out = model(x)  # torch.Size([1024, 6, 1655])
        y = crop_target_to_match(y, out.shape[2]).long()

        loss, dice_loss, total_loss = get_loss(out, y, 0.01, 1e-4)

        total_loss.backward()
        optimizer.step()
        epoch_loss.append(loss.detach().cpu().item())
        epoch_dice_loss.append(dice_loss.detach().cpu().item())
        epoch_total_loss.append(total_loss.detach().cpu().item())

        dice, miou, pixel_acc = get_performance(out, y, args.n_classes, smooth=1e-4)
        epoch_dice_score.append(dice)
        epoch_miou.append(miou)
        epoch_pixel_acc.append(pixel_acc)

    epoch_train_loss = np.mean(np.array(epoch_loss))
    epoch_train_dice_loss = np.mean(np.array(epoch_dice_loss))
    epoch_train_total_loss = np.mean(np.array(epoch_total_loss))

    epoch_dice_score = np.mean(np.array(epoch_dice_score))
    epoch_miou = np.mean(np.array(epoch_miou))
    epoch_pixel_acc = np.mean(np.array(epoch_pixel_acc))

    model.eval()
    epoch_test_loss, epoch_test_dice_loss, epoch_test_total_loss = [], [], []
    epoch_test_dice, epoch_test_miou, epoch_test_pixel_acc = [], [], []
    for batch in eval_dataloader:
        x, y = batch
        x = x.permute(0, 2, 1)
        x, y = x.to(device), y.to(device)
        y = y.squeeze()

        out = model(x)  # torch.Size([1024, 6, 1655])
        # out = torch.argmax(out, dim=1)  # point 별 최빈값을 predict로 출력. torch.Size([1024, 1655])
        y = crop_target_to_match(y, out.shape[2]).long()

        loss, dice_loss, total_loss = get_loss(out, y, 0.01, 1e-4)
        dice, miou, pixel_acc = get_performance(out, y, args.n_classes, 1e-4)

        epoch_test_loss.append(loss.detach().cpu().item())
        epoch_test_dice_loss.append(dice_loss.detach().cpu().item())
        epoch_test_total_loss.append(total_loss.detach().cpu().item())

        epoch_test_dice.append(dice)
        epoch_test_miou.append(miou)
        epoch_test_pixel_acc.append(pixel_acc)

    epoch_test_loss = np.mean(np.array(epoch_test_loss))
    epoch_test_dice_loss = np.mean(np.array(epoch_test_dice_loss))
    epoch_test_total_loss = np.mean(np.array(epoch_test_total_loss))

    epoch_test_dice = np.mean(np.array(epoch_test_dice))
    epoch_test_miou = np.mean(np.array(epoch_test_miou))
    epoch_test_pixel_acc = np.mean(np.array(epoch_test_pixel_acc))

    # print('[Epoch] : {0:03d} \t '
    #       '[Train CE Loss] : {1:.4f} \t '
    #       '[Eval CE Loss] : {2:.4f} \t '
    #       '[Train Dice Loss] : {3:.4f} \t '
    #       '[Eval Dice Loss] : {4:.4f} \t '
    #       '[Train Total Loss] : {5:.4f} \t '
    #       '[Eval Total Loss] : {6:.4f} \t '.format(epoch + 1,
    #                                               epoch_train_loss,
    #                                               epoch_test_loss,
    #                                               epoch_train_dice_loss,
    #                                               epoch_test_dice_loss,
    #                                               epoch_train_total_loss,
    #                                                epoch_test_total_loss))

    print('[Epoch] : {0:03d} \t '
          '[Train Total Loss] : {1:.4f} \t '
          '[Eval Total Loss] : {2:.4f} \t '
          '[mPA] : {3:.4f} \t '
          '[mIoU] : {4:.4f} \t '
          '[Dice] : {5:.4f} \t '.format(epoch + 1,
                                        epoch_train_total_loss,
                                        epoch_test_total_loss,
                                        epoch_test_pixel_acc,
                                        epoch_test_miou,
                                        epoch_test_dice))


    writer.add_scalar('Train/Loss/Epoch', epoch_train_total_loss, epoch+1)
    writer.add_scalar('Eval/Loss/Epoch', epoch_test_total_loss, epoch+1)
    writer.add_scalar('Eval/mPA/Epoch', epoch_test_pixel_acc, epoch+1)
    writer.add_scalar('Eval/mIoU/Epoch', epoch_test_miou, epoch+1)
    writer.add_scalar('Eval/Dice/Epoch', epoch_test_dice, epoch+1)

writer.close()
