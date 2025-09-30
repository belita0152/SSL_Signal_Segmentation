import os
import torch
import numpy as np
from unet import UNet

import pickle
from torch.utils.data import DataLoader, Dataset
import argparse


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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


args = get_args()
tuning_dataset = TupleDataset(tuning_x, tuning_y)
tuning_dataloader = DataLoader(tuning_dataset, batch_size=args.batch_size, shuffle=True)
eval_dataset = TupleDataset(eval_x, eval_y)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)

def crop_target_to_match(y, target_length):
    total_length = y.shape[1]  # e.g., 3000
    start = (total_length - target_length) // 2
    end = start + target_length
    return y[:, start:end]


def get_iou_score(pred, real, n_classes):
    iou_per_class = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(n_classes):
        pred_c = (pred == cls).float()
        real_c = (real == cls).float()
        # pred_c = pred[:, cls, :]  # [B, L]
        # real_c = real[:, :]  # [B, L]

        intersection = (pred_c * real_c).sum()
        union = pred_c.sum() + real_c.sum() - intersection

        if union == 0:
            iou_score = torch.tensor(float('nan'), device=pred.device)
        else:
            iou_score = intersection / union

        iou_per_class.append(iou_score.detach().cpu().item())

    miou_score = np.nanmean(np.array(iou_per_class))

    return miou_score


def evaluate_encoder(encoder_path, dataloader, num_classes=6):
    model = UNet(num_classes=num_classes, encoder_ckpt=encoder_path).to(device)
    model.eval()

    total_miou = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.permute(0, 2, 1).to(device)  # [B, C, L]
            y = y.squeeze().to(device).long()  # [B, L]

            out = model(x)
            y = crop_target_to_match(y, out.shape[2])  # 너의 함수

            miou = get_iou_score(out, y, num_classes)
            total_miou.append(miou)

    return np.mean(total_miou)


def find_best_encoder(checkpoint_dir):
    best_miou = -1
    best_path = None

    for i in range(1, 201, 1):
        path = os.path.join(checkpoint_dir, f'VGG_encoder_epoch{i}.pth')
        miou = evaluate_encoder(path, eval_dataloader)
        print(f"[{os.path.basename(path)}] mIoU: {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            best_path = path

        print(f"\n Best encoder: {os.path.basename(best_path)} | mIoU: {best_miou:.4f}")
    return best_path


if __name__ == "__main__":
    checkpoint_dir = '/home/lhy/SSL_Signal_Segmentation/pretrained/checkpoint_VGG_enc'
    best_encoder_path = find_best_encoder(checkpoint_dir)


# Result => Best encoder: VGG_encoder_epoch82.pth
