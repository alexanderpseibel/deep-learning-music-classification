# src/models/crnn_fma.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    """
    CRNN model inspired by:
    'Convolutional Recurrent Neural Networks for Music Classification'
    Choi et al., 2017.
    """

    def __init__(self, num_classes, cnn_channels=[32, 64, 128, 256], rnn_hidden=128, rnn_layers=2):
        super().__init__()

        # ---------- CNN Layers ----------
        def cnn_block(in_c, out_c, pool):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ELU(),
                nn.MaxPool2d(pool)
            )

        # Four CNN blocks (from paper Fig. 1(d))
        self.block1 = cnn_block(1,   cnn_channels[0], (2, 2))
        self.block2 = cnn_block(cnn_channels[0], cnn_channels[1], (3, 3))
        self.block3 = cnn_block(cnn_channels[1], cnn_channels[2], (4, 4))
        self.block4 = cnn_block(cnn_channels[2], cnn_channels[3], (4, 4))

        # ---------- GRU (2 layers) ----------
        self.gru = nn.GRU(
            input_size=cnn_channels[3],
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=False
        )

        # Final classifier
        self.fc = nn.Linear(rnn_hidden, num_classes)

    def forward(self, x):
        # x: (B, 1, mel, time)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # reshape for GRU
        # current shape: (B, C, Freq=1, Time=T)
        B, C, F, T = x.shape
        x = x.squeeze(2)          # → (B, C, T)
        x = x.permute(0, 2, 1)    # → (B, T, C)

        # RNN
        out, h = self.gru(x)      # h: (layers, B, hidden)
        h_last = h[-1]            # last layer hidden state

        return self.fc(h_last)
