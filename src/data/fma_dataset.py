import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class FMAAudioDataset(Dataset):
    """
    Loads mel spectrogram + multi-hot labels.
    Ensures fixed time dimension via pad/crop.
    """

    TARGET_T = 3000   #

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_cols = [c for c in df.columns if c.startswith("label_")]

    def __len__(self):
        return len(self.df)

    def _fix_length(self, mel):
        """
        Crop or pad mel to TARGET_T.
        mel shape = (128, T)
        """
        _, T = mel.shape

        if T > self.TARGET_T:
            # crop center
            start = (T - self.TARGET_T) // 2
            mel = mel[:, start:start+self.TARGET_T]

        elif T < self.TARGET_T:
            # pad end with zeros
            pad_amount = self.TARGET_T - T
            mel = np.pad(mel, ((0, 0), (0, pad_amount)), mode='constant')

        return mel

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load mel
        mel = np.load(row["mel_path"])  # (128, T)

        mel = self._fix_length(mel)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        # Load labels
        labels = torch.tensor(
            row[self.label_cols].values.astype(np.float32)
        )

        if self.transform:
            mel = self.transform(mel)

        return mel, labels

