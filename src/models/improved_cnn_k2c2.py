# src/models/improved_cnn_k2c2.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedK2C2CNN(nn.Module):
    """
    A Pytorch implementation of the k2c2 CNN architecture
    from Choi et al., 2017 (Music Tagging) ─ best pure CNN.

    Features:
    - 5 convolutional blocks
    - Each block: Conv → BN → ELU → Conv → BN → ELU → Pool
    - Pooling schedule matches the paper:
        (2×4) → (2×4) → (2×4) → (3×5) → (4×4)
    - Fully convolutional (no dense layers)
    - Ends with global average pooling + 1×1 classifier
    """

    def __init__(self, num_classes):
        super().__init__()

        # Conv block builder
        def conv_block(in_ch, out_ch, pool_kernel):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ELU(),

                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ELU(),

                nn.Dropout2d(0.2),

                nn.MaxPool2d(pool_kernel)
            )

        # Architecture depth follows k2c2 style: 5 conv blocks
        self.block1 = conv_block(1,   32, (2, 4))
        self.block2 = conv_block(32,  64, (2, 4))
        self.block3 = conv_block(64, 128, (2, 4))
        self.block4 = conv_block(128, 256, (3, 5))
        self.block5 = conv_block(256, 512, (4, 4))

        # Global average pooling → (B, 512, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 1×1 classifier (fully convolutional)
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.block1(x)   # (B, 32, 64, ~750)
        x = self.block2(x)   # (B, 64, 32, ~185)
        x = self.block3(x)   # (B, 128, 16, ~46)
        x = self.block4(x)   # (B, 256, 5, ~8)
        x = self.block5(x)   # (B, 512, 1, 1)

        x = self.gap(x)      # (B, 512, 1, 1)
        x = self.classifier(x)  # (B, num_classes, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # → (B, num_classes)

        return x
