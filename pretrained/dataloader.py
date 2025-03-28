import glob
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

base_path = os.path.join(os.getcwd(), 'shhs2_o')
data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535
data_split_ratio = {'SSL': 0.7, 'Tuning': 0.2, 'Eval': 0.1}


class SleepDataset(Dataset):
    def __init__(self, mode):
        # 전체 데이터셋
        from parser import Segment
        total_x, total_y = Segment().parser(data_path, mask_path)
        self.inputs = total_x
        self.labels = total_y

          # Split train/tuning/eval dataset
        tuning_ratio = data_split_ratio['Tuning'] / (data_split_ratio['Tuning'] + data_split_ratio['Eval'])
        eval_ratio = data_split_ratio['Eval'] / (data_split_ratio['Tuning'] + data_split_ratio['Eval'])
        
        x_pretrained, x_finetuned, y_pretrained, y_finetuned = train_test_split(
            self.inputs, self.labels, train_size=data_split_ratio['SSL'], random_state=777
        )
        
        x_tuning, x_eval, y_tuning, y_eval = train_test_split(
            x_finetuned, y_finetuned, train_size=tuning_ratio, random_state=777
        )

        # __init__에서 mode를 이용해 data/label 직접 지정 => 뒤에서 추가로 나눌 필요 없음
        if mode == 'train':
            self.inputs = x_pretrained
            self.labels = y_pretrained
        elif mode == 'tuning':
            self.inputs = x_tuning
            self.labels = y_tuning
        elif mode == 'eval':
            self.inputs = x_eval
            self.labels = y_eval

    def __len__(self):
        return len(self.inputs)  # x의 크기인지, y의 크기인지?

    def __getitem__(self, item):  # 샘플마다 데이터 로드
        x, y = self.inputs[item], self.labels[item]  # 각 sample 데이터와 label 반환
        return x, y


if __name__ == '__main__':
    import os
    import glob
    import pandas as pd
    base_path = os.path.join(os.getcwd(), '..', 'shhs2_o')
    data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
    mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535

    from parser import Segment
    parser = Segment()
    x1, y1 = parser.parser(data_path, mask_path)
    print(x1.shape)  # (243207, 3000, 4)
    print(y1.shape)  # (243207, 3000, 4)

    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = torch.tensor(x1, dtype=torch.float32)
    train_dataloader = DataLoader(data, batch_size=64, shuffle=True)

