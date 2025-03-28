import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class Segment(object):
    def __init__(self):
        self.ch_names = ['AIRFLOW', 'THOR RES', 'ABDO RES', 'SaO2']
        self.event_name = ['Obstructive apnea',  'Central apnea', 'Mixed apnea',  'Hypopnea', 'Arousal', 'SpO2 artifact']
        self.second = 300  # segmentation size
        self.sfreq = 10  # sampling rate
        self.base_path = os.path.join(os.getcwd(), 'shhs2_o')
        self.data_path = sorted(glob.glob(os.path.join(self.base_path, '**/*data.parquet')))  # len: 2535
        self.mask_path = sorted(glob.glob(os.path.join(self.base_path, '**/*mask.parquet')))  # len: 2535

    def parser(self, data_path, mask_path):
        total_x, total_y = [], []
        for x_path, y_path in zip(data_path, mask_path):
            x = pd.read_parquet(x_path).T  # ex) (258000, 4) -> (4, 258000)
            y = pd.read_parquet(y_path).T  # ex) (258000, 1) -> (1, 258000)

            x, y = np.array(x), np.array(y)  # pd.DF -> np.array

            x = StandardScaler().fit_transform(x)  # 데이터 scaling하는 이유: outlier의 영향 최소화 (mne의 robust scaler를 사용할 것.)
            x = x.reshape(-1, self.second * self.sfreq, len(self.ch_names))
            y = y.reshape(-1, self.second * self.sfreq, 1)
            total_x.append(x)
            total_y.append(y)
        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)
        return total_x, total_y


# 일단 augmentation 코드 작성한 후, 필요한 형태로 코드 작성하자.
class ToTensor(object):  # Transform class
    pass


if __name__ == "__main__":
  base_path = os.path.join(os.getcwd(), '..', 'shhs2_o')
  data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
  mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535
  
  parser = Segment()
  x, y = parser.parser(data_path, mask_path)
  print(x.shape)  # (243207, 3000, 4)
  print(y.shape)  # (243207, 3000, 1)
