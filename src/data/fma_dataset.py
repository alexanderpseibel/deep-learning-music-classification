# src/data/fma_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset


class FMAAudioDataset(Dataset):
    """
    Loads mel spectrogram + multi-hot labels.
    Ensures fixed time dimension via center-crop or zero-pad.
    """

    TARGET_T = 3000  # frames at 32kHz / 10ms hop ≈ 30s

    def __init__(self, df, mean=None, std=None):
        self.df = df.reset_index(drop=True)
        self.label_cols = [c for c in df.columns if c.startswith("label_")]
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df)

    def _fix_length(self, mel):
        _, T = mel.shape
        if T > self.TARGET_T:
            start = (T - self.TARGET_T) // 2
            mel = mel[:, start:start + self.TARGET_T]
        elif T < self.TARGET_T:
            mel = np.pad(mel, ((0, 0), (0, self.TARGET_T - T)), mode="constant")
        return mel

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        mel = np.load(row["mel_path"])
        mel = self._fix_length(mel)

        if self.mean is not None and self.std is not None:
            mel = (mel - self.mean) / self.std

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, 128, T)
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))

        return mel, labels
