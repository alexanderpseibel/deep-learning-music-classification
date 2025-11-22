#src/models/improved_k2c2_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedK2C2CNN(nn.Module):
    """
    Tier-1 improved version of K2C2:
    - MaxPool replaced with strided conv for learnable downsampling
    - ELU replaced with GELU for better gradient flow
    - Architecture remains fully convolutional (pure CNN)
    """

    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_ch, out_ch, stride_hw):
            """
            One block:
            Conv → BN → GELU → Conv(strided) → BN → GELU → Dropout
            """
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),

                # strided conv replaces MaxPool
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride_hw, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),

                nn.Dropout2d(0.2),
            )

        # Use same downsampling schedule as original K2C2
        self.block1 = conv_block(1,   32, (2, 4))
        self.block2 = conv_block(32,  64, (2, 4))
        self.block3 = conv_block(64, 128, (2, 4))
        self.block4 = conv_block(128, 256, (3, 5))
        self.block5 = conv_block(256, 512, (4, 4))

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 1×1 classifier
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.gap(x)
        x = self.classifier(x)
        x = x.squeeze(-1).squeeze(-1)
        return x
