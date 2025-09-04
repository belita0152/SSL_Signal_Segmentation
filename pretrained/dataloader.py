from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import mne

import warnings
warnings.filterwarnings(action='ignore')

class TorchDataset(Dataset):  # Parsing + ToTensor
    def __init__(self, data_dir, mask_dir, transform=None):
        self.ch_names = ['AIRFLOW', 'THOR RES', 'ABDO RES', 'SaO2']
        self.event_name = ['Obstructive apnea',  'Central apnea', 'Mixed apnea',
                           'Hypopnea', 'Arousal', 'SpO2 artifact']
        self.second = 300  # segmentation size
        self.sfreq = 10  # sampling rate

        self.data_path = data_dir  # len: 2535
        self.mask_path = mask_dir  # len: 2535
        self.split = {'SSL': 0.7, 'Tuning': 0.2, 'Eval': 0.1}

        self.data_x, self.data_y = self.parser(self.data_path, self.mask_path)

    def parser(self, data_dir, mask_dir):
        total_x, total_y = [], []
        for x_path, y_path in zip(data_dir, mask_dir):
            x = pd.read_parquet(x_path).T  # ex) (258000, 4) -> (4, 258000)
            y = pd.read_parquet(y_path).T  # ex) (258000, 1) -> (1, 258000)

            x, y = np.array(x), np.array(y)  # pd.DF -> np.array
            x = x.reshape(-1, self.second * self.sfreq, len(self.ch_names))
            y = y.reshape(-1, self.second * self.sfreq, 1)

            x = np.swapaxes(x, 1, 2)  # (N, 4, 3000)
            info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='resp')
            x = mne.decoding.Scaler(info, scalings='median').fit_transform(x)  # 데이터 scaling하는 이유: outlier의 영향 최소화 (mne의 robust scaler를 사용할 것.)
            total_x.append(x)

            y = y.squeeze()
            # y = (y != 0).astype(np.float32, copy=False)  # binary
            total_y.append(y)  # (N, 3000, 1) -> (N, 3000)

        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)
        return total_x, total_y  # 전체 파싱한 데이터셋

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, item):  # 샘플마다 데이터 로드
        x = torch.tensor(self.data_x[item])
        y = torch.tensor(self.data_y[item], dtype=torch.float32)

        return x, y


if __name__ == '__main__':
    import os
    import glob

    split_ratio = {'Train': 0.65, 'Val': 0.05}
    base_path = os.path.join(os.getcwd(), '../../../../data/segmentation/shhs2_o')
    data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
    mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535

    # 2. Split into train / tuning / evaluation partitions
    train_size = int(len(data_path) * split_ratio['Train'])
    train_data_path = data_path[0:train_size]
    train_mask_path = mask_path[0:train_size]

    val_size = int(len(data_path) * split_ratio['Val'])
    val_data_path = data_path[train_size:(train_size + val_size)]
    val_mask_path = mask_path[train_size:(train_size + val_size)]

    train_data = TorchDataset(data_dir=train_data_path, mask_dir=train_mask_path, transform=None)
    val_data = TorchDataset(data_dir=val_data_path, mask_dir=val_mask_path, transform=None)
    print(train_data.data_x.shape, train_data.data_y.shape)

    import matplotlib.pyplot as plt
    data = train_data.data_y
    plt.plot(data[1000])
    plt.show()