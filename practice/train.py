import warnings
import numpy as np
import torch
import random
import argparse
from typing import Tuple, Dict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from Segmenter.segmenter import Segmenter
from pretrained.data_loader import get_dataset
from utils.loss import CrossEntropyDiceLoss
import utils.metric as metric
import json

warnings.filterwarnings("ignore")

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
ckpt_dir = './checkpoint_VGG_250902'
log_dir = './log_250902'

writer = SummaryWriter(log_dir=log_dir)

def get_args():
    parser = argparse.ArgumentParser(description='SSL_Signal_Segmentation')

    # Dataset
    parser.add_argument('--dataset_name', default='ahi', choices=['heartbeat', 'ahi'])
    parser.add_argument('--save_path', type=str, default=os.path.join('..', 'result'))
    parser.add_argument('--flag', type=bool, default=False)

    # Train Hyperparameter
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--save_checkpoint', default=True, type=bool)
    parser.add_argument('--n_channels', default=4, type=int)
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.999, type=float)
    parser.add_argument('--device', default=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--seed', default=42, type=int)

    return parser.parse_args()

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(batch_x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batch_x.to(torch.float32, non_blocking=True).to(device)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Model
        self.model: nn.Module = Segmenter(
            n_cls=args.n_classes,
            patch_size=(1, 10),
            flag=args.flag
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.criterion = CrossEntropyDiceLoss()
        self.scaler = torch.cuda.amp.GradScaler()

        # Data Loader
        self.train_loader, self.eval_loader = self._build_dataloaders()

    def _make_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if args.dataset_name == 'heartbeat':
            x = torch.stack([data['ECG_1'], data['ECG_2']], dim=1)  # (B, 2, T)
        elif args.dataset_name == 'ahi':
            x = torch.stack([data['AIRFLOW'], data['THOR RES'], data['ABDO RES'], data["SaO2"]], dim=1)  # (B, 4, T)
        else:
            raise ValueError(f"Invalid dataset name provided: {args.dataset_name}")
        return to_device(x, self.device)

    def train_one_epoch(self, epoch: int):
        self.model.train()
        for data, target in self.train_loader:
            self.optimizer.zero_grad()

            x = self._make_input(data)
            y = target.long().to(self.device)
            y[y > 0] = 1        # To binary classification -> Arrhythmia Detection

            with torch.cuda.amp.autocast():
                logits = self.model(x)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.scheduler.step(epoch=epoch)

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int) -> Dict:
        self.model.eval()
        preds_all, reals_all = [], []

        for data, target in self.eval_loader:
            x = self._make_input(data)
            y = target.long().to(self.device)
            if args.dataset_name == 'heartbeat':
                y[y > 0] = 1        # To binary classification
            elif args.dataset_name == 'ahi':
                y[y > 0] = 1  # To binary classification
                # y[(1 <= y) & (y <= 3)] = 1
                # y[y == 4] = 2
                # y[y == 5] = 3

            logits = self.model(x)
            preds = logits.argmax(dim=1)

            preds_all.append(preds.detach().cpu())
            reals_all.append(y.detach().cpu())

        y_pred = torch.cat(preds_all, dim=0).reshape(-1).numpy()
        y_true = torch.cat(reals_all, dim=0).reshape(-1).numpy()

        result = metric.calculate_segmentation_metrics(y_pred, y_true, num_classes=2)
        accuracy, iou_macro, dice_macro = result['accuracy'], result['iou_macro'], result['dice_macro']
        print(f'[Epoch]: {epoch:03d} => '
              f'[Accuracy] : {accuracy*100:.2f} [IoU Macro] : {iou_macro*100:.2f} [Dice Macro] : {dice_macro*100:.2f}')
        return result


    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset, eval_dataset = get_dataset(name=self.args.dataset_name)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, eval_loader

    def run(self):
        results = []
        for epoch in range(1, self.args.epochs + 1):
            self.train_one_epoch(epoch)
            result = self.eval_one_epoch(epoch)
            results.append(result)

        file_path = os.path.join(args.save_path, args.dataset_name, 'segmenter.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    Trainer(args).run()