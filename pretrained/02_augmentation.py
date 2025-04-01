import random
import numpy as np
import torch

import warnings
warnings.filterwarnings(action='ignore')

class SigAugmentation(object):
    def __init__(self):
        self.n_permutation = 5
        self.second = 300
        self.sampling_rate = 10
        self.input_length = self.second * self.sampling_rate
        self.gn_scaling = list(np.arange(0.05, 0.10, 0.01))
        self.augmentations = ['random_crop', 'random_gaussian_noise',
                              'random_temporal_cutout', 'random_permutation']

    def process(self, x: torch.Tensor, aug_name, p=0.5):
        import copy
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
        else:
            return x

    def random_crop(self, x: torch.Tensor, p=0.5):  # x = inputs
        # https://arxiv.org/pdf/2109.07839 - Crop & resize
        sfreq_crop = 5  # 정수로 나와야 idx로 계산 가능. 전체 segment의 50% 크기로 cropping

        if random.random() < p:
            index_1 = torch.randint(low=0, high=sfreq_crop * self.second, size=(1,), device=x.device)[0].item()  # 크기 1의 정수 생성 (인덱스로 기능)
            index_2 = int(index_1 + self.input_length * 0.5)
            # 1. Crop
            x_split = x[:, :, index_1:index_2]  # (4, 3000) -> (4, 1500)

            # 2. Resize to original segmentation size (= second * sfreq = 3000)
            x_split = torch.nn.functional.interpolate(
                x_split,
                size=self.input_length,
                mode='linear',
                align_corners=False
            )

        else:  # random.random() > p 에 해당하면 적용하지 않고, 바로 데이터 추가
            x_split = x

        return x_split

    def random_gaussian_noise(self, x: torch.Tensor, p=0.5):
        mu = 0.0

        if random.random() < p:
            std = torch.tensor(self.gn_scaling, device=x.device).multinomial(num_samples=1)[0] * torch.std(x)
            noise = torch.normal(mu, std, size=x.shape, device=x.device)
            x_split = x + noise  # 원본 신호에 노이즈 추가
            # print(x_split.shape)

        else:
            x_split = x
            # print(x_split.shape)

        return x_split

    def random_temporal_cutout(self, x: torch.Tensor, p=0.5):  # temporal cutout
        start_list = torch.arange(0, self.sampling_rate * self.second, device=x.device)  # start point to crop
        width_list = torch.arange(int(self.sampling_rate / 4), int(self.sampling_rate / 2),
                                  device=x.device)  # cropped width

        if random.random() < p:
            start = start_list[torch.randint(0, len(start_list), (1,), device=x.device)].item()  # 무작위로 start point 1개 선택
            width = width_list[torch.randint(0, len(width_list), (1,), device=x.device)].item()  # 무작위로 width 1개 선택
            print(start, width)

            # 평균값 계산
            mean_value = x.mean().item()

            # 선택된 부분을 평균값으로 대체
            indices = torch.arange(start, start+width, device=x.device).long()  # 선택한 부분의 index 범위
            indices = indices[indices < x.shape[0]]
            x_split = x.index_fill(0, indices, mean_value)

        else:
            x_split = x

        return x_split

    def random_permutation(self, x: torch.Tensor, p=0.5):
        """Randomly segment, shuffle, and merge the signals."""

        # 1. Segment
        if random.random() < p:
            indexes = torch.randperm(self.input_length)[:self.n_permutation - 1]
            indexes = torch.cat([torch.tensor([0]), indexes, torch.tensor([self.input_length])])
            indexes, _ = torch.sort(indexes)  # 자를 index 랜덤하게 고르고, 오름차순으로 정렬

            # for문 활용, 해당 indexes에 맞게 데이터를 segment -> samples 리스트에 segment들 저장
            segments = []
            for idx_1, idx_2 in zip(indexes[:-1], indexes[1:]):
                segments.append(x[:, :, idx_1:idx_2])  # time points에 대해 segment 나누기

            # 2. Shuffle
            # 데이터 조각의 인덱스를 np.random.permutation
            perm_indices = torch.randperm(self.n_permutation)
            shuffled_segments = [segments[idx] for idx in perm_indices]  #순열된 인덱스에 따라 samples에서 데이터 조각 선택 -> shuffled_segments에 추가

            # 3. Merge
            x_split = torch.cat(shuffled_segments, dim=-1)  # 조각들 합치기

        else:  # random.random() > p 에 해당한다면 순열 적용하지 않고, 바로 데이터 추가
            x_split = x

        return x_split

    def convert_augmentation(self, x: torch.Tensor):
        aug_1, aug_2 = random.sample(self.augmentations, 2)
        x1 = self.process(x, aug_name=aug_1, p=0.95)
        x2 = self.process(x, aug_name=aug_2, p=0.95)
        return x1, x2

#
# import torch
# aug = SigAugmentation()
#
# xx = aug.process(
#     x=torch.randn(size=(63, 3000)),
#     aug_name='random_crop'
# )
# print(xx.shape)
# from dataloader import TorchDataset
# from torch.utils.data import DataLoader, random_split
# dataset = TorchDataset(transform=None)
# train_data, tuning_data, eval_data = random_split(dataset,
#                                                   [dataset.split['SSL'],
#                                                    dataset.split['Tuning'],
#                                                    dataset.split['Eval']])  # ratio or length


# import pickle
# import copy
# with open('train_data.pkl', 'rb') as f:
#     train_data = pickle.load(f)
#
# train_data = copy.deepcopy(train_data)
# from encoder import TupleDataset
# from torch.utils.data import DataLoader
# train_data = TupleDataset(train_data)
# train_dataloader = DataLoader(train_data, shuffle=True, batch_size=4, drop_last=True)
#
# sig_aug = SigAugmentation()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for batch, _ in train_dataloader:  # batch: torch.Size([64, 4, 3000])
#     x = batch.to(device)
#     aug = sig_aug.random_permutation(batch, p=0.95)
#     print(aug.shape)


# # [1] Random crop
# x1 = augment.random_crop(x0)
# print(x1.shape)
#
# # [2] Gaussian Noise (Jitter) : 원본 신호에 노이즈 추가
# x2 = augment.random_gaussian_noise(x0)
# print(x2.shape)
#
# # [3] Random Permutation
# x3 = augment.random_permutation(x1)
# print(x3.shape)
#
# # [4] Temporal Cutout: 일부 cut -> 자른 부분은 평균으로 대체
# x4 = augment.random_temporal_cutout(x1)
# print(x4.shape)
