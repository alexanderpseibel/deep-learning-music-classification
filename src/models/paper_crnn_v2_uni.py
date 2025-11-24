# src/models/paper_crnn_v2_uni.py

import torch
import torch.nn as nn


class PaperCRNNv2Uni(nn.Module):
    """
    Paper-style CRNN (Choi et al., 2017), corrected for arbitrary mel bins.
    Uses UNIDIRECTIONAL GRU.
    """

    def __init__(self, num_classes):
        super().__init__()

        # --- CNN FRONTEND (4 layers) ---
        def conv_block(in_ch, out_ch, pool_kernel=(2, 2)):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ELU(),
                nn.MaxPool2d(pool_kernel),
            )

        self.conv1 = conv_block(1,   96, (2, 2))
        self.conv2 = conv_block(96, 128, (2, 2))
        self.conv3 = conv_block(128, 128, (2, 2))
        self.conv4 = conv_block(128, 128, (2, 2))

        # Weak dropout between conv blocks (paper)
        self.drop = nn.Dropout(0.1)

        # --- SAFE FREQUENCY COLLAPSE ---
        # Collapses (freq_dim) → 1 while keeping time dimension unchanged
        self.freq_collapse = nn.AdaptiveAvgPool2d((1, None))

        # --- RNN backend (UNIDIRECTIONAL) ---
        self.gru = nn.GRU(
            input_size=128,      # matches C after conv frontend
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # CNN
        x = self.drop(self.conv1(x))
        x = self.drop(self.conv2(x))
        x = self.drop(self.conv3(x))
        x = self.drop(self.conv4(x))

        # Collapse frequency dim → (B, C, 1, T)
        x = self.freq_collapse(x)

        # Reshape for GRU → (B, T, C)
        x = x.squeeze(2).transpose(1, 2)

        # GRU
        _, h_last = self.gru(x)   # (num_layers, B, H)
        h_last = h_last[-1]       # last layer

        return self.fc(h_last)
