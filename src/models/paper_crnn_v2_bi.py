# src/models/paper_crnn_v2_bi.py

import torch
import torch.nn as nn


class PaperCRNNv2Bi(nn.Module):
    """
    Paper-style CRNN with BIDIRECTIONAL GRU.
    Output dimension = 2 × hidden_size.
    """

    def __init__(self, num_classes):
        super().__init__()

        # --- CNN FRONTEND ---
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

        self.drop = nn.Dropout(0.1)

        # SAFE COLLAPSE
        self.freq_collapse = nn.AdaptiveAvgPool2d((1, None))

        # --- Bidirectional GRU ---
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Output size = 256×2 (bidirectional)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.drop(self.conv1(x))
        x = self.drop(self.conv2(x))
        x = self.drop(self.conv3(x))
        x = self.drop(self.conv4(x))

        x = self.freq_collapse(x)
        x = x.squeeze(2).transpose(1, 2)

        # h_last dims:
        #   (num_layers * 2, B, hidden_size)
        _, h_last = self.gru(x)

        # take last layer forward + backward
        h_last = torch.cat([h_last[-2], h_last[-1]], dim=-1)

        return self.fc(h_last)
