# src/data/fma_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class FMAAudioDataset(Dataset):
    """
    Loads mel spectrogram + multi-hot labels.
    """

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_cols = [c for c in df.columns if c.startswith("label_")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load mel spectrogram (128 × T)
        mel = np.load(row["mel_path"])
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, 128, T)

        # Load multi-hot label
        labels = row[self.label_cols].values.astype(np.float32)
        labels = torch.tensor(labels)

        if self.transform:
            mel = self.transform(mel)

        return mel, labels
