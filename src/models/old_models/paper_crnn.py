import torch
import torch.nn as nn


class PaperCRNN(nn.Module):
    """
    Paper-exact CRNN (Choi et al., 2017),
    adapted to 128×3000 input but otherwise identical.
    """

    def __init__(self, num_classes):
        super().__init__()

        # --- CNN FRONTEND (4 layers) ---
        def conv_block(in_ch, out_ch, pool_kernel):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ELU(),
                nn.MaxPool2d(pool_kernel),
            )

        self.conv1 = conv_block(1,   96, (2, 2))
        self.conv2 = conv_block(96, 128, (3, 3))
        self.conv3 = conv_block(128, 128, (4, 4))
        self.conv4 = conv_block(128, 128, (4, 4))

        # Weak dropout between conv blocks (paper)
        self.drop = nn.Dropout(0.1)

        # --- RNN backend ---
        self.gru = nn.GRU(
            input_size=128,   # freq collapsed
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.drop(self.conv1(x))
        x = self.drop(self.conv2(x))
        x = self.drop(self.conv3(x))
        x = self.drop(self.conv4(x))

        # x: (B, C, Freq=1, Time=T)
        x = x.squeeze(2)  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)

        _, h_last = self.gru(x)  # h_last: (num_layers, B, H)
        h_last = h_last[-1]      # final layer's hidden state

        return self.fc(h_last)
