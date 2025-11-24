#src/models/paper_crnn_uni.py
import torch
import torch.nn as nn


class CRNNv2Uni(nn.Module):
    """
    Improved CRNN (unidirectional GRU)
    - GELU instead of ELU
    - Consistent (2×2) pooling schedule to keep more time resolution
    - GRU(256) with dropout=0.3 to prevent overfitting
    - Same output interface as your existing models
    """

    def __init__(self, num_classes):
        super().__init__()

        # --- CNN FRONTEND (4 conv blocks) ---
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.MaxPool2d((2, 2)),     # keep more time resolution
            )

        self.conv1 = conv_block(1,   96)
        self.conv2 = conv_block(96, 128)
        self.conv3 = conv_block(128, 128)
        self.conv4 = conv_block(128, 128)

        # small spatial dropout
        self.drop = nn.Dropout(0.1)

        # GRU backend
        self.gru = nn.GRU(
            input_size=128,    # after frequency collapse
            hidden_size=256,
            num_layers=2,
            dropout=0.3,       # dropout inside GRU
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.drop(self.conv1(x))
        x = self.drop(self.conv2(x))
        x = self.drop(self.conv3(x))
        x = self.drop(self.conv4(x))

        # (B, C, F, T) → collapse freq dim
        x = x.squeeze(2)         # (B, C, T)
        x = x.transpose(1, 2)    # (B, T, C)

        _, h_last = self.gru(x)
        h_last = h_last[-1]      # final layer

        return self.fc(h_last)
