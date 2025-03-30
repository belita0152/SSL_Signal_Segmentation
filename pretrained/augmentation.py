import random
import numpy as np
import copy
from tslearn.preprocessing import TimeSeriesResampler

import warnings
warnings.filterwarnings(action='ignore')

class SigAugmentation(object):
    def __init__(self):
        self.n_permutation = 5
        self.second = 300
        self.sampling_rate = 10
        self.input_length = self.second * self.sampling_rate
        self.gn_scaling = list(np.arange(0.05, 0.10, 0.01))

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
        else:
            return x

    def random_crop(self, x, p=0.5):  # x = inputs
        # https://arxiv.org/pdf/2109.07839 - Crop & resize
        sfreq_crop = 5  # 정수로 나와야 idx로 계산 가능. 전체 segment의 50% 크기로 cropping
        x = x.dataset[x.indices][0]  # (170245, 3000, 4)

        new_x = []
        for x_split in x:  # x_split = input data per subject
            x_split = np.array(x_split)  # (3000, 4)
            if random.random() < p:
                index_1 = np.random.randint(low=0, high=sfreq_crop * self.second, size=1)[0]  # 크기 1의 정수 생성 (인덱스로 기능)
                index_2 = int(index_1 + self.input_length * 0.5)
                # 1. Crop
                x_split = x_split[index_1:index_2, :]  # (3000, 4) -> (1500, 4)
                print(x_split.shape)
                # print(x_split.shape)
                # 2. Resize to original segmentation size (= second * sfreq = 3000)
                x_split = TimeSeriesResampler(sz=self.input_length).fit_transform(x_split)
                new_x.append(x_split)
            else:  # random.random() > p 에 해당하면 적용하지 않고, 바로 데이터 추가
                new_x.append(x_split)
        new_x = np.array(new_x)
        return new_x

    def random_gaussian_noise(self, x, p=0.5):
        x = x.dataset[x.indices][0]  # (170245, 3000, 4)

        mu = 0.0
        new_x = []
        for x_split in x:
            x_split = np.array(x_split)
            if random.random() < p:
                std = np.random.choice(self.gn_scaling, 1)[0] * np.std(x_split)
                noise = np.random.normal(mu, std, x_split.shape)
                x_split = x_split + noise  # 원본 신호에 노이즈 추가
                print(x_split.shape)
                new_x.append(x_split)
            else:
                new_x.append(x_split)
                print(x_split.shape)
        new_x = np.array(new_x)
        return new_x

    def random_temporal_cutout(self, x, p=0.5):  # temporal cutout
        x = x.dataset[x.indices][0]  # (170245, 3000, 4)

        new_x = []
        start_list = list(np.arange(0, self.sampling_rate * self.second))  # start point to crop
        width_list = list(np.arange(int(self.sampling_rate / 4), int(self.sampling_rate / 2)))  # cropped width

        for x_split in x:
            x_split = np.array(x_split)
            if random.random() < p:
                start = np.random.choice(start_list, 1)[0]  # 무작위로 start point 1개 선택
                width = np.random.choice(width_list, 1)[0]  # 무작위로 width 1개 선택
                np.put(x_split, np.arange(start, start+width), x_split.mean())  # start부터 start+width 시점까지 x_split의 평균으로 대체
                new_x.append(x_split)
            else:
                new_x.append(x_split)
        new_x = np.array(new_x)
        return new_x

    def random_permutation(self, x, p=0.5):
        """Randomly segment, shuffle, and merge the signals."""
        # 데이터 준비 #
        x = x.dataset[x.indices][0]  # (170245, 3000, 4)
        x = np.array(x)
        aug_x = []

        # 1. Segment
        for sample in x:
            if random.random() < p:
                indexes = list(np.random.choice(self.input_length, self.n_permutation - 1, replace=False))
                indexes += [0, self.input_length]
                indexes = list(np.sort(indexes))

                # for문 활용, 해당 indexes에 맞게 데이터를 segment -> samples 리스트에 segment들 저장
                segments = []
                for idx_1, idx_2 in zip(indexes[:-1], indexes[1:]):
                    segments.append(sample[:, idx_1:idx_2])

                # 2. Shuffle
                # 데이터 조각의 인덱스를 np.random.permutation
                shuffled_segments = []
                for idx in np.random.permutation(np.arange(self.n_permutation)):
                    # 순열된 인덱스에 따라 samples에서 데이터 조각 선택 -> shuffled_segments에 추가
                    shuffled_segments.append(segments[idx])

                # 3. Merge
                # np.concatenate 를 활용해 전체 합치기 (axis=-1)
                shuffled_segments = np.concatenate(shuffled_segments, axis=-1)
                print(shuffled_segments.shape)
                aug_x.append(shuffled_segments)

            else:  # random.random() > p 에 해당한다면 순열 적용하지 않고, 바로 데이터 추가
                aug_x.append(sample)
                print(sample.shape)

        # 데이터 반환
        aug_x = np.array(aug_x)  # np.array 형태로 바꿔서 출력
        return aug_x


