import os
import glob
import pandas as pd

# Path
base_path = os.path.join(os.getcwd(), '..', 'shhs2_o')  # 현재 경로에 따라 path 수정
data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535

#######################################################################

# Parsing : segment 만들기
from parser import Segment
parser = Segment()
x1, y1 = parser.parser(data_path, mask_path)
print(x1.shape)  # (243207, 3000, 4)
print(y1.shape)  # (243207, 3000, 1)

# # 전체 x: (243207, 3000, 4), y: (243207, 3000, 1)
dataset = TorchDataset(transform=None)
train_data, tuning_data, eval_data = random_split(dataset,
                                                  [dataset.split['SSL'],
                                                   dataset.split['Tuning'],
                                                   dataset.split['Eval']])  # ratio or length
x_train = train_data.dataset[train_data.indices] # 현재 Tuple. [0] : x, [1] : y

print("Train: ", len(train_data))  # 170245
print("Tuning: ", len(tuning_data))  # 48642
print("Eval: ", len(eval_data))  # 24320

print(train_dataloader)
print(tuning_dataloader)
print(eval_dataloader)

#######################################################################

# Augmentation : 4가지 기법 중 랜덤하게 2가지 선택
augment = SigAugmentation()
from dataloader import *
x0 = train_data  # (170245, 3000, 4) - from dataloader

# [1] Random crop
x1 = augment.random_crop(x0)
print(x1.shape)

# [2] Gaussian Noise (Jitter) : 원본 신호에 노이즈 추가
x2 = augment.random_gaussian_noise(x0)
print(x2.shape)

# [3] Random Permutation
x3 = augment.random_permutation(x1)
print(x3.shape)

# [4] Temporal Cutout: 일부 cut -> 자른 부분은 평균으로 대체
x4 = augment.random_temporal_cutout(x1)
print(x4.shape)

#######################################################################

# Encoder -> DataLoader, Encoder block별 shape 확인
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

import pickle
import copy
with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

train_data = copy.deepcopy(train_data)

train_data = TupleDataset(train_data)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=64, drop_last=True)

train_features, train_labels = next(iter(train_dataloader))
print(len(train_data))        # 170245
print(len(train_dataloader))  # 2660
print(f"Feature batch shape: {train_features.size()}")  # torch.Size([64, 4, 3000])
print(f"Labels batch shape: {train_labels.size()}")  # torch.Size([64, 1, 3000])

model = VGG(in_channels=4, num_classes=6)
model.to(device)

train_features = train_features.to(device)
model_output = model(train_features)
print(model_output)
print(model_output.shape)
