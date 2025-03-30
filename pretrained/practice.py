import os
import glob
import pandas as pd

# Path
base_path = os.path.join(os.getcwd(), '..', 'shhs2_o')  # 현재 경로에 따라 path 수정
data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535

# Parsing : segment 만들기
from parser import Segment
parser = Segment()
x1, y1 = parser.parser(data_path, mask_path)
print(x1.shape)  # (243207, 3000, 4)
print(y1.shape)  # (243207, 3000, 1)

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
