# src/data/transforms.py
import numpy as np
import random

class SpecAugment:
    """
    Applies SpecAugment: time masking + frequency masking.
    Expects mel spectrogram as numpy array (128, T).
    """

    def __init__(self, freq_masks=2, time_masks=2,
                 freq_max_width=20, time_max_width=50):
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_max_width = freq_max_width
        self.time_max_width = time_max_width

    def __call__(self, mel):
        mel = mel.copy()
        n_mels, n_steps = mel.shape

        # ----- Frequency masking -----
        for _ in range(self.freq_masks):
            f = random.randint(0, self.freq_max_width)
            f0 = random.randint(0, max(0, n_mels - f))
            mel[f0:f0+f, :] = 0

        # ----- Time masking -----
        for _ in range(self.time_masks):
            t = random.randint(0, self.time_max_width)
            t0 = random.randint(0, max(0, n_steps - t))
            mel[:, t0:t0+t] = 0

        return mel
