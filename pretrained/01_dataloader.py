from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import *
import numpy as np
import torch

import warnings
warnings.filterwarnings(action='ignore')

class TorchDataset(Dataset):  # Parsing + ToTensor
    def __init__(self, transform=None):
        self.ch_names = ['AIRFLOW', 'THOR RES', 'ABDO RES', 'SaO2']
        self.event_name = ['Obstructive apnea',  'Central apnea', 'Mixed apnea',  'Hypopnea', 'Arousal', 'SpO2 artifact']
        self.second = 300  # segmentation size
        self.sfreq = 10  # sampling rate

        self.data_path = data_path  # len: 2535
        self.mask_path = mask_path  # len: 2535
        self.split = {'SSL': 0.7, 'Tuning': 0.2, 'Eval': 0.1}


        self.data_x, self.data_y = self.parser(self.data_path, self.mask_path)

    def parser(self, data_dir, mask_dir):
        total_x, total_y = [], []
        for x_path, y_path in zip(data_dir, mask_dir):
            x = pd.read_parquet(x_path).T  # ex) (258000, 4) -> (4, 258000)
            y = pd.read_parquet(y_path).T  # ex) (258000, 1) -> (1, 258000)

            x, y = np.array(x), np.array(y)  # pd.DF -> np.array

            x = StandardScaler().fit_transform(x)  # 데이터 scaling하는 이유: outlier의 영향 최소화 (mne의 robust scaler를 사용할 것.)
            x = x.reshape(-1, self.second * self.sfreq, len(self.ch_names))
            y = y.reshape(-1, self.second * self.sfreq, 1)
            total_x.append(x)
            total_y.append(y)
        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)

        return total_x, total_y  # 전체 파싱한 데이터셋

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, item):  # 샘플마다 데이터 로드
        x = torch.tensor(self.data_x[item], dtype=torch.float)
        y = torch.tensor(self.data_y[item], dtype=torch.float)

        x = x.transpose(0, 1)  # (time, channel) -> (channel, time)로 변경
        y = y.transpose(0, 1)
        return x, y


# # 전체 x: (243207, 3000, 4), y: (243207, 3000, 1)
# dataset = TorchDataset(transform=None)
# train_data, tuning_data, eval_data = random_split(dataset,
#                                                   [dataset.split['SSL'],
#                                                    dataset.split['Tuning'],
#                                                    dataset.split['Eval']])  # ratio or length
# x_train = train_data.dataset[train_data.indices] # [0] : x, [1] : y

# print("Train: ", len(train_data))  # 170245
# print("Tuning: ", len(tuning_data))  # 48642
# print("Eval: ", len(eval_data))  # 24320
#
# print(train_dataloader)
# print(tuning_dataloader)
# print(eval_dataloader)


import os
import glob

# 0. Data Path
base_path = os.path.join(os.getcwd(), '..', 'shhs2_o')
data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535

# 1. Split data/mask path
split_ratio = {'SSL': 0.7, 'Tuning': 0.2, 'Eval': 0.1}
dataset = len(data_path)
n_train = int(dataset * split_ratio['SSL'])
n_tuning = int(dataset * split_ratio['Tuning'])
n_eval = dataset - n_train - n_tuning

train_data_path = data_path[0:n_train]
tuning_data_path = data_path[n_train:(n_train+n_tuning)]
eval_data_path = data_path[(n_train+n_tuning):]

train_mask_path = mask_path[0:n_train]
tuning_mask_path = mask_path[n_train:(n_train+n_tuning)]
eval_mask_path = mask_path[(n_train+n_tuning):]

