# -*- coding:utf-8 -*-
import os
import numpy as np
from typing import List


def sliding_window_1d(
    data: np.ndarray,
    fs: int,
    window_sec: float,
    step_sec: float,
) -> np.ndarray:
    data = np.asarray(data).squeeze()
    assert data.ndim == 1, f"data must be 1D, got shape={data.shape}"

    window = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    n = data.shape[0]

    if window <= 0 or step <= 0:
        raise ValueError(f"window/step must be positive. window={window}, step={step}")
    if n < window:
        return np.empty((0, window), dtype=data.dtype)

    num_windows = 1 + (n - window) // step
    sw = np.lib.stride_tricks.sliding_window_view(data, window)[::step]
    sw = sw[:num_windows]
    return sw


def list_subdirs(base_path: str) -> List[str]:
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found or not a directory: {base_path}")
    subdirs = [os.path.join(base_path, d)
               for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d))]
    subdirs.sort()
    return subdirs


def get_dataset(name: str):
    name = name.lower()

    if name == "heartbeat":
        from data.data_loader import HeartbeatDataset
        channel_num = 2     # [ECG1, ECG2]
        class_num = 2       # [0: normal heartbeat, 1: abnormal heartbeat]

        train_dataset = HeartbeatDataset(
            base_path='/data/segmentation/mit_bit',
            fs=125,
            second=5.0,
            sliding_window_sec=5.0,
            down_sampling=False,
            train_ratio=0.8,
            train=True
        )
        eval_dataset = HeartbeatDataset(
            base_path='/data/segmentation/mit_bit',
            fs=125,
            second=5.0,
            sliding_window_sec=5.0,
            down_sampling=False,
            train_ratio=0.8,
            train=False
        )
        return (train_dataset, eval_dataset), (channel_num, class_num)

    elif name == "ahi":
        from data.data_loader import AHIDataset
        channel_num = 3     # [AIRFLOW, THOR RES, ABDO RES]
        class_num = 3       # [0: normal, 1: apnea, 2: hypopnea]

        train_dataset = AHIDataset(
            base_path='/data/segmentation/shhs2_o',
            fs=10,
            second=300,
            sliding_window_sec=300,
            train_ratio=0.8,
            train=True,
            down_sampling=True
        )
        eval_dataset = AHIDataset(
            base_path='/data/segmentation/shhs2_o',
            fs=10,
            second=300,
            sliding_window_sec=300,
            train_ratio=0.8,
            train=False,
        )
        return (train_dataset, eval_dataset), (channel_num, class_num)

    elif name == "gesture":
        from data.data_loader import NinaproDataset
        channel_num = 12          # [EMG_1, EMG_2, EMG_3, ... ]
        class_num = 5             # [0: normal, 1: not normal]

        train_dataset = NinaproDataset(
            base_path='/data/segmentation/ninapro_f3',
            fs=500,
            second=50,
            sliding_window_sec=50,
            train_ratio=0.8,
            train=True,
            down_sampling=False
        )
        eval_dataset = NinaproDataset(
            base_path='/data/segmentation/ninapro_f3',
            fs=500,
            second=50,
            sliding_window_sec=50,
            train_ratio=0.8,
            train=False,
        )
        return (train_dataset, eval_dataset), (channel_num, class_num)

    elif name == 'heartsound':
        from data.data_loader import HeartSoundDataset
        channel_num = 1    # [HeartSound]
        class_num = 4      # [0: s1, 1: systole, 2: s2, 3: diastole]
        train_dataset = HeartSoundDataset(
            base_path='/data/segmentation/heartsound_f2',
            fs=500,
            second=20,
            sliding_window_sec=10,
            train_ratio=0.8,
            train=True,
            down_sampling=False
        )
        eval_dataset = HeartSoundDataset(
            base_path='/data/segmentation/heartsound_f2',
            fs=500,
            second=20,
            sliding_window_sec=20,
            train_ratio=0.8,
            train=False,
        )
        return (train_dataset, eval_dataset), (channel_num, class_num)
    else:
        raise ValueError(f"Invalid dataset name provided: {name}")
