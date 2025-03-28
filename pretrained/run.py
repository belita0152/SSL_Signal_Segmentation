"""

Step 1. Encoder 학습시키기

"""

import os
import glob
import pandas as pd

# Path
base_path = os.path.join(os.getcwd(), '..', 'shhs2_o')
data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535

# Parsing : segment 만들기
from parser import Segment
parser = Segment()
x1, y1 = parser.parser(data_path, mask_path)
print(x1.shape)  # (243207, 3000, 4)
print(y1.shape)  # (243207, 3000, 1)

# Augmentation : 4가지 기법 중 랜덤하게 2가지 선택
# [1] Random crop
augment = SigAugmentation(data_path, mask_path)
x2 = augment.random_crop(x1)
print(x2.shape)
