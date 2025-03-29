"""

Step 1. Encoder 학습시키기

"""

import warnings
import numpy as np
import torch
import random
import argparse
import os
import glob
from dataloader import Dataset
from torch.utils.data import DataLoader, random_split
import tqdm

warnings.filterwarnings("ignore")

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser(description='Sleep Signal Segmentation')
    # Dataset
    parser.add_argument('--base_path', default=os.path.join(os.getcwd(), 'shhs2_o'))
    # base_path 중 data.parquet => train data, mask.parquet => mask (label)
    parser.add_argument('--sfreq',  default=100, type=int)
    parser.add_argument('--data_split_ratio', default={'SSL': 0.7, 'Tuning': 0.2, 'Eval': 0.1}, type=float)

    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=30, type=int)
    parser.add_argument('--train_base_learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--save_checkpoint', default=True, type=bool)
    parser.add_argument('--n_channels', default=4, type=int)
    parser.add_argument('--n_classes', default=6, type=int)

    return parser.parse_args()


# class Trainer(object):
#     def __init__(self, args):
#         self.args = args
#         self.model = UNet(backbone=VGGNet_encoder, )
#         self.batch_size =
#         self.lr =
#         self.optimizer =
#         self.scheduler =
#         self.tensorboard_path =


def train_pretrained_model(
        model,  # VGGNet Encoder
        device,  # dont know what it is
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        train_percent: float = augments.data_split_ratio['SSL'],
        tuning_percent: float = augments.data_split_ratio['Tuning'],
        eval_percent: float = augments.data_split_ratio['Eval'],
        save_checkpoint: bool = True,
        sig_scale: float = 0.5,  # dont know what it is
        amp: bool = False,  # dont know what it is
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    train_dataset = Dataset(data_path, mask_path)

    # 2. Split into train / tuning / evaluation partitions
    n_train = int(len(train_dataset) * train_percent)
    n_tuning = int(len(train_dataset) * tuning_percent)
    n_eval = int(len(train_dataset) * eval_percent)
    train_set, tuning_set, eval_set = random_split(train_dataset, [n_train, n_tuning, n_eval], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)  # don't know why use this code
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    tuning_loader = DataLoader(tuning_set, shuffle=True, **loader_args)
    eval_loader = DataLoader(eval_set, shuffle=False, drop_last=True,  **loader_args)

    # logging, initialize logging -> don't know why use these types of codes

    # 4. Set up the optimizer, loss, learning rate scheduler and loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)  # foreach -> don't know properly
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # really don't know this code
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)  # don't know what this means
    ############################# 3/22 NEED TO WORK
    criterion = SimCLR()  # put train_loader => in the SimCLR codes, I have to make augmented signal x (using data_path)
    global_step = 0  # don't know why global_step needs

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit=) as pbar:  # unit=img -> signal
            for batch in train_loader:
                signals, true_masks = batch['signal'], batch['mask']

                # images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                # true_masks = true_masks.to(device=device, dtype=torch.long)
                #
                # masks_pred = model(images)
                # loss = criterion(masks_pred, true_masks)  # compare with UNet github codes
                # loss += dice_loss(
                #     F.softmax(masks_pred, dim=1).float(),
                #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float,
                #     multiclass=True
                # )

                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

        # 6. Save checkpoints
        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)  # find parameters
            state_dict=model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved.')  # don't know what this code means


if __name__ == '__main__':
    args = get_args()
    print(args.data_split_ratio['SSL'])  # 0.7
    print(args.data_split_ratio['Tuning'])  # 0.2
    print(args.data_split_ratio['Eval'])  # 0.1

    dir_sig = glob.glob(os.path.join(args.base_path, '*/data.parquet'), recursive=True)  # len: 2535
    dir_mask = glob.glob(os.path.join(args.base_path, '*/mask.parquet'), recursive=True)  # len: 2535
    dir_checkpoint = os.path.join('..', 'checkpoints')

    # logging, device, logging.info => compare with UNet github codes

    model = UNet(n_channels=args.n_channels, n_classes=args.n_classes)
    model.to(device=device)
    train_pretrained_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.lr,
        device=device,
        amp=args.amp
    )

    # trainer = Trainer(augments)  => 1. class 면 -> argument 지정해서
    # trainer.train()  => class 안쪽 함수 불러와야 함
    train_pretrained_model(args)  # 2. 함수 면 -> 바로 불러올 수 있음
