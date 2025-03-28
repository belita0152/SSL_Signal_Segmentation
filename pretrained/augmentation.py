import random
import numpy as np
import copy
from tslearn.preprocessing import TimeSeriesResampler
import os
import glob
import pandas as pd

class SigAugmentation(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_permutation = 5
        self.second = 300
        self.sampling_rate = 10
        self.input_length = self.second * self.sampling_rate

    def process(self, x, aug_name, p=0.5):
        x = copy.deepcopy(x)
        if aug_name == 'random_crop':
            x = self.random_crop(x, p)
            return x
        elif aug_name == 'random_gaussian_noise':
            x = self.random_gaussian_noise(x, p)
            return x
        elif aug_name == 'random_permutation':
            x = self.random_permutation(x, p)
            return x
        elif aug_name == 'random_temporal_cutout':
            x = self.random_temporal_cutout(x, p)
            return x

    def random_crop(self, x, p=0.5):  # x = inputs
        # https://arxiv.org/pdf/2109.07839 - Crop & resize
        sfreq_crop = 3  # 정수로 나와야 idx로 계산 가능. 전체 segment의 70% 크기로 cropping

        new_x = []
        for x_split in x:  # x_split = input data per subject
            print(x_split)
            if random.random() < p:
                index_1 = np.random.randint(low=0, high=sfreq_crop * self.second, size=1)[0]  # 크기 1의 정수 생성 (인덱스로 기능)
                index_2 = index_1 + self.input_length * 0.7
                # 1. Crop
                x_split = x_split.iloc[:, index_1:index_2]  # pd.DataFrame -> using iloc
                # 2. Resize to original segmentation size (= second * sfreq = 3000)
                x_split = TimeSeriesResampler(sz=self.input_length).fit_transform(x_split)
                # x_split = np.squeeze(x_split, axis=-1)
                new_x.append(x_split)
            else:  # random.random() > p 에 해당하면 적용하지 않고, 바로 데이터 추가
                new_x.append(x_split)
        new_x = np.array(new_x)
        return new_x


if __name__ == "__main__":
  base_path = os.path.join(os.getcwd(), '..', 'shhs2_o')
  data_path = sorted(glob.glob(os.path.join(base_path, '**/*data.parquet')))  # len: 2535
  mask_path = sorted(glob.glob(os.path.join(base_path, '**/*mask.parquet')))  # len: 2535
  
  from parser import Segment
  parser = Segment()
  
  x1, y1 = parser.parser(data_path, mask_path)
  print(x1.shape)
  print(y1.shape)
  
  augment = SigAugmentation(data_path, mask_path)
  x2 = augment.random_crop(x1)
  print(x2.shape)

