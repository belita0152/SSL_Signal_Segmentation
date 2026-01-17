# -*- coding:utf-8 -*-
import copy
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from data import utils
from torch.utils.data import Dataset, DataLoader


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        signal_cols: Sequence[str],
        *,
        down_sampling: False,
        train: bool = True,
        train_ratio: float = 0.8,
        fs: int = 125,
        window_sec: float = 10.0,
        step_sec: float = 10.0,
    ):
        super().__init__()
        self.base_path = base_path
        self.signal_cols = list(signal_cols)
        self.fs = fs
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.train = train
        self.down_sampling = down_sampling

        self.paths = utils.list_subdirs(base_path)
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No subdirectories under: {base_path}")

        split = int(len(self.paths) * float(train_ratio))
        self.paths_train = self.paths[:split]
        self.paths_test = self.paths[split:]

        target_paths = self.paths_train if train else self.paths_test

        self.data_dict, self.mask_arr = self._load_and_window_all(target_paths)

    def _load_one(self, path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        data_path = os.path.join(path, "data.parquet")
        mask_path = os.path.join(path, "mask.parquet")

        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Missing data.parquet at: {path}")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Missing mask.parquet at: {path}")

        data_df = pl.read_parquet(data_path)
        mask_df = pl.read_parquet(mask_path)

        for col in self.signal_cols:
            if col not in data_df.columns:
                raise KeyError(f"Column '{col}' not found in {data_path}. "
                               f"Available: {data_df.columns}")

        if "MASK" not in mask_df.columns:
            raise KeyError(f"'MASK' column not found in {mask_path}. "
                           f"Available: {mask_df.columns}")
        mask_1d = mask_df["MASK"].to_numpy().squeeze()
        signal_1d = {col: data_df[col].to_numpy().squeeze() for col in self.signal_cols}
        return signal_1d, mask_1d

    def _window_one(self, sig_1d: Dict[str, np.ndarray], mask_1d: np.ndarray
                    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

        win_signals = {
            col: utils.sliding_window_1d(arr, self.fs, self.window_sec, self.step_sec)
            for col, arr in sig_1d.items()
        }
        win_mask = utils.sliding_window_1d(mask_1d, self.fs, self.window_sec, self.step_sec)

        lengths = [v.shape[0] for v in win_signals.values()] + [win_mask.shape[0]]
        n = min(lengths) if lengths else 0
        if n == 0:
            w = int(round(self.window_sec * self.fs))
            win_signals = {k: np.empty((0, w), dtype=float) for k in self.signal_cols}
            win_mask = np.empty((0, w), dtype=float)
        else:
            win_signals = {k: v[:n] for k, v in win_signals.items()}
            win_mask = win_mask[:n]
        return win_signals, win_mask

    @staticmethod
    def _downsample_windows(
        sig_w: Dict[str, np.ndarray],
        m_w: np.ndarray,
        *,
        neg_ratio: float = 0.5,   # 양성 대비 음성(완전 0 윈도우) 유지 비율
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        n = m_w.shape[0]
        flat = m_w.reshape(n, -1).astype(bool)

        pos_idx = flat.any(axis=1)
        neg_idx = ~pos_idx

        num_pos = int(pos_idx.sum())
        keep_idx = pos_idx.copy()

        if num_pos > 0:
            max_negs = int(neg_ratio * num_pos)
            neg_candidates = np.flatnonzero(neg_idx)
            if len(neg_candidates) > 0 and max_negs > 0:
                keep_negs = np.random.choice(
                    neg_candidates,
                    size=min(max_negs, len(neg_candidates)),
                    replace=False,
                )
                keep_idx[keep_negs] = True
        else:
            neg_candidates = np.flatnonzero(neg_idx)
            keep_count = max(1, len(neg_candidates) // 20)  # 5% 정도
            if keep_count > 0:
                keep_negs = np.random.choice(neg_candidates, size=keep_count, replace=False)
                keep_idx[keep_negs] = True

        filtered_sig_w = {k: sig_w[k][keep_idx] for k in sig_w}
        filtered_m_w = m_w[keep_idx]
        return filtered_sig_w, filtered_m_w, keep_idx

    def _load_and_window_all(self, paths: Sequence[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        data_chunks: Dict[str, List[np.ndarray]] = {k: [] for k in self.signal_cols}
        mask_chunks: List[np.ndarray] = []

        for p in paths:
            sig_1d, m_1d = self._load_one(p)
            sig_w, m_w = self._window_one(sig_1d, m_1d)

            if self.down_sampling:
                m_w_expend = np.sum(m_w, axis=-1).astype(np.bool_)
                for k in self.signal_cols:
                    data_chunks[k].append(sig_w[k][m_w_expend])
                mask_chunks.append(m_w[m_w_expend])
            else:
                for k in self.signal_cols:
                    data_chunks[k].append(sig_w[k])
                mask_chunks.append(m_w)

        cat_dict = {k: np.concatenate(v, axis=0) if len(v) else np.empty((0, 0), dtype=float)
                    for k, v in data_chunks.items()}
        cat_mask = np.concatenate(mask_chunks, axis=0) if len(mask_chunks) else np.empty((0, 0), dtype=float)
        return cat_dict, cat_mask

    # ---- Dataset API ----
    def __len__(self) -> int:
        return self.mask_arr.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        data_np = {k: self.data_dict[k][idx] for k in self.signal_cols}
        mask_np = self.mask_arr[idx]

        data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data_np.items()}
        mask = torch.tensor(mask_np, dtype=torch.long)
        return data, mask


# -----------------------------------------------------------------------------
# Concrete Datasets
# -----------------------------------------------------------------------------
class HeartbeatDataset(SlidingWindowDataset):
    """MIT-BIH Arrhythmia Database"""
    def __init__(
        self,
        base_path: str = "/data/segmentation/mit_bit",
        train: bool = True,
        fs: int = 125,
        second: float = 30.0,
        down_sampling: bool = False,
        sliding_window_sec: float = 30.0,
        train_ratio: float = 0.8,
    ):
        super().__init__(
            base_path=base_path,
            signal_cols=("ECG_1", "ECG_2"),
            train=train,
            train_ratio=train_ratio,
            fs=fs,
            down_sampling=down_sampling,
            window_sec=second,
            step_sec=sliding_window_sec,
        )

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        data_np = {k: self.data_dict[k][idx] for k in self.signal_cols}
        mask_np = self.mask_arr[idx]
        mask_np[mask_np > 0] = 1        # [Normal Beat, Anomaly Beat]

        data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data_np.items()}
        mask = torch.tensor(mask_np, dtype=torch.long)
        return data, mask


class AHIDataset(SlidingWindowDataset):
    """Sleep Heart Health Study"""
    def __init__(
        self,
        base_path: str = "/data/segmentation/shhs",
        train: bool = True,
        fs: int = 125,
        second: float = 10.0,
        down_sampling: bool = False,
        sliding_window_sec: float = 10.0,
        train_ratio: float = 0.8,
    ):
        super().__init__(
            base_path=base_path,
            signal_cols=("AIRFLOW", "THOR RES", "ABDO RES"),
            down_sampling=down_sampling,
            train=train,
            train_ratio=train_ratio,
            fs=fs,
            window_sec=second,
            step_sec=sliding_window_sec,
        )
        self.channel_num = 3
        self.class_num = 3

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        data_np = {k: self.data_dict[k][idx] for k in self.signal_cols}
        mask_np = self.mask_arr[idx]
        mask_np[(1 <= mask_np) & (mask_np <= 3)] = 1
        mask_np[mask_np == 4] = 2
        mask_np[mask_np == 5] = 0
        mask_np[mask_np == 6] = 0
        mask_np[mask_np == 7] = 0        # [Normal, Apnea, Hypopnea]

        data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data_np.items()}
        mask = torch.tensor(mask_np, dtype=torch.long)
        return data, mask


class NinaproDataset(SlidingWindowDataset):
    # https://ninapro.hevs.ch/instructions/DB2.html
    def __init__(
        self,
        base_path: str = "/data/segmentation/ninapro_o",
        train: bool = True,
        fs: int = 2000,
        second: float = 1,
        down_sampling: bool = False,
        sliding_window_sec: float = 1,
        train_ratio: float = 0.8,
    ):
        super().__init__(
            base_path=base_path,
            # signal_cols=("emg_1", 'emg_2'),
            signal_cols=("emg_1", "emg_2", "emg_3", "emg_4",
                         "emg_5", "emg_6", "emg_7", "emg_8",
                         "emg_9", "emg_10", "emg_11", "emg_12"),
            down_sampling=down_sampling,
            train=train,
            train_ratio=train_ratio,
            fs=fs,
            window_sec=second,
            step_sec=sliding_window_sec,
        )
        self.channel_num = 12
        self.class_num = 5

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        New Class 0: 휴식 (Rest)
         - Original: 0

        New Class 1: 손 펴기 (Hand Open)
         - Original: 6 (Hand open)

        New Class 2: 손 쥐기 (Hand Close / Fist)
         - Original: 5 (Hand close), 22, 23, 24, 25 (다양한 직경의 파워 쥐기)

        New Class 3: 손가락 집기 (Pinch)
         - Original: 7, 8, 9, 10 (각 손가락과 엄지 집기), 18, 19, 20, 21, 32 (기능적인 집기 동작)

        New Class 4: 손목 움직임 (Wrist Movement)
         - Original: 12, 13, 14 (굽힘, 회전 등)
        """
        data_np = {k: self.data_dict[k][idx] for k in self.signal_cols}
        mask_np = self.mask_arr[idx]

        # mask_np[mask_np > 0] = 1    # TODO : 0 / 1
        # -----------------------------
        hand_open_labels = [6]
        hand_close_labels = [5, 22, 23, 24, 25]
        pinch_labels = [7, 8, 9, 10, 18, 19, 20, 21, 32]
        wrist_labels = [12, 13, 14]

        conditions = [
            np.isin(mask_np, hand_open_labels),
            np.isin(mask_np, hand_close_labels),
            np.isin(mask_np, pinch_labels),
            np.isin(mask_np, wrist_labels),
        ]

        choices = [1, 2, 3, 4]
        y_remapped = np.select(conditions, choices, default=-1)
        y_remapped[mask_np == 0] = 0  # 휴식 클래스(0)는 확실하게 0으로 지정

        # mask_np[mask_np == 4] = 2
        # mask_np[mask_np == 5] = 0
        # mask_np[mask_np == 6] = 0
        # mask_np[mask_np == 7] = 0
        mask_np = y_remapped

        data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data_np.items()}
        mask = torch.tensor(mask_np, dtype=torch.long)
        return data, mask



class HeartSoundDataset(SlidingWindowDataset):
    def __init__(
            self,
            base_path: str = "/data/segmentation/heartsound_f2",
            train: bool = True,
            fs: int = 500,
            second: float = 10,
            down_sampling: bool = False,
            sliding_window_sec: float = 1,
            train_ratio: float = 0.8,
    ):
        super().__init__(
            base_path=base_path,
            signal_cols=("heart_sound", ),
            down_sampling=down_sampling,
            train=train,
            train_ratio=train_ratio,
            fs=fs,
            window_sec=second,
            step_sec=sliding_window_sec,
        )
        self.channel_num = 3
        self.class_num = 4  # [0: s1, 1: systole, 2: s2, 3: diastole]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        data_np = {k: self.data_dict[k][idx] for k in self.signal_cols}
        mask_np = self.mask_arr[idx] - 1

        data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data_np.items()}
        mask = torch.tensor(mask_np, dtype=torch.long)
        return data, mask


if __name__ == "__main__":
    ds = HeartSoundDataset()
    for d, m in ds:
        print(set(m.tolist()))

    # ds2 = HeartbeatDataset(
    #     base_path="/data/segmentation/mit_bit",
    #     train=False,
    #     fs=125,
    #     second=5.0,
    #     sliding_window_sec=5.0,
    #     train_ratio=0.8,
    # )
    # print(ds2.mask_arr.shape)
    #
