#src/models/paper_crnn_v2_bi.py
import torch
import torch.nn as nn


class CRNNv2Bi(nn.Module):
    """
    Improved CRNN (bidirectional GRU)
    - GELU activations
    - (2×2) pooling schedule
    - GRU(256×2) with dropout=0.3
    """

    def __init__(self, num_classes):
        super().__init__()

        # --- CNN FRONTEND ---
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.MaxPool2d((2, 2)),
            )

        self.conv1 = conv_block(1,   96)
        self.conv2 = conv_block(96, 128)
        self.conv3 = conv_block(128, 128)
        self.conv4 = conv_block(128, 128)

        self.drop = nn.Dropout(0.1)

        # bidirectional GRU
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
            bidirectional=True  # <---
        )

        # bidir doubles hidden dim → 512
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.drop(self.conv1(x))
        x = self.drop(self.conv2(x))
        x = self.drop(self.conv3(x))
        x = self.drop(self.conv4(x))

        x = x.squeeze(2)         # (B, C, T)
        x = x.transpose(1, 2)    # (B, T, C)

        _, h_last = self.gru(x)
        h_last = h_last[-1]      # last GRU layer (BiGRU returns both directions)

        return self.fc(h_last)
