# -*- coding:utf-8 -*-
import os
import json
import argparse
import warnings
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import metric
from exp.unet.model import UNet1D
from data.utils import get_dataset
from utils.loss import CrossEntropyDiceLoss
from utils.utils import save_best_model, save_experiment_results, print_best_results

warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch_x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batch_x.to(torch.float32, non_blocking=True).to(device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='gesture', choices=['heartbeat', 'ahi', 'gesture', 'heartsound'])
    parser.add_argument('--save_path', type=str, default=os.path.join('..', 'result'))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--ce_dice_alpha', default=0.7, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)

    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self.save_path = os.path.join(args.save_path, args.dataset_name)

        # Data Loader
        (self.train_loader, self.eval_loader), (self.channel_num, self.class_num) = self._build_dataloaders()

        # Model
        self.model_cfg = {
            'in_channels': self.channel_num,
            'out_channels': self.class_num,
            'stem_channels': 32,
            'stage_channels': (32, 64, 128, 128),
            'stage_blocks': (2, 2, 2, 1),
            'stem_kernel': 50,  # Hz / 4
            'block_kernel': 25,  # Hz / 8
        }

        self.model: nn.Module = UNet1D(
            **self.model_cfg,
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = CrossEntropyDiceLoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def _make_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.stack(list(data.values()), dim=1)
        return to_device(x, self.device)

    def train_one_epoch(self, epoch: int):
        self.model.train()
        for data, target in self.train_loader:
            self.optimizer.zero_grad()

            x = self._make_input(data)
            y = target.long().to(self.device)

            with torch.cuda.amp.autocast():
                logits = self.model(x)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int) -> Dict:
        self.model.eval()
        preds_all, reals_all = [], []

        for data, target in self.eval_loader:
            x = self._make_input(data)
            y = target.long().to(self.device)

            logits = self.model(x)
            preds = logits.argmax(dim=1)

            preds_all.append(preds.detach().cpu())
            reals_all.append(y.detach().cpu())

        y_pred = torch.cat(preds_all, dim=0).reshape(-1).numpy()
        y_true = torch.cat(reals_all, dim=0).reshape(-1).numpy()

        result = metric.calculate_segmentation_metrics(y_pred, y_true, num_classes=self.class_num)
        accuracy, iou_macro, dice_macro = result['accuracy'], result['iou_macro'], result['dice_macro']
        print(f'[Epoch]: {epoch:03d} => '
              f'[Accuracy] : {accuracy*100:.2f} [IoU Macro] : {iou_macro*100:.2f} [Dice Macro] : {dice_macro*100:.2f}')
        return result

    def _build_dataloaders(self) -> Tuple[Tuple[DataLoader, DataLoader], Tuple[int, int]]:
        (train_dataset, eval_dataset), (channel_num, class_num) = get_dataset(name=self.args.dataset_name)
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
        return (train_loader, eval_loader), (channel_num, class_num)

    def run(self):
        results = []
        best_iou = 0.0
        best_overall_stats = {
            'epoch': 0,
            'accuracy': 0.0,
            'iou_macro': 0.0,
            'dice_macro': 0.0
        }

        for epoch in range(1, self.args.epochs + 1):
            self.train_one_epoch(epoch)
            result = self.eval_one_epoch(epoch)
            results.append(result)

            if result['iou_macro'] > best_iou:
                best_iou = result['iou_macro']
                best_overall_stats = result
                save_best_model(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    iou=best_iou,
                    save_dir=self.save_path,
                    model_name='segmenter'
                )

        # Save hyperparameters
        save_experiment_results(
            args=self.args,
            results=results,
            best_iou=best_iou,
            model_config=self.model_cfg,  # settings
            save_dir=self.save_path,
            model_name='segmenter'
        )

        print_best_results(best_overall_stats)


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    Trainer(args).run()
