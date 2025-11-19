# src/data/fma_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class FMAAudioDataset(Dataset):
    """
    Loads mel spectrogram + multi-hot labels.
    Ensures fixed time dimension via pad/crop.
    """

    TARGET_T = 3000  # fixed mel width

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_cols = [c for c in df.columns if c.startswith("label_")]

    def __len__(self):
        return len(self.df)

    def _fix_length(self, mel):
        _, T = mel.shape

        if T > self.TARGET_T:
            start = (T - self.TARGET_T) // 2
            mel = mel[:, start:start + self.TARGET_T]

        elif T < self.TARGET_T:
            pad_amount = self.TARGET_T - T
            mel = np.pad(mel, ((0, 0), (0, pad_amount)), mode='constant')

        return mel

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load mel spectrogram
        mel = np.load(row["mel_path"])
        mel = self._fix_length(mel)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1,128,T)

        # Load multi-hot labels
        labels = torch.tensor(
            row[self.label_cols].values.astype(np.float32)
        )

        # Apply transforms if any
        if self.transform:
            mel = self.transform(mel)

        # Return ONLY mel + labels
        return mel, labels
