# src/models/k2c2_silu_depthwise_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise → Pointwise convolution block.
    Same output shape as a regular Conv2d but MUCH cheaper.
    """
    def __init__(self, ch):
        super().__init__()
        self.depthwise = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.pointwise = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class K2C2_SiLU_DepthwiseCNN(nn.Module):
    """
    Original K2C2 architecture, but with:
    - SiLU activation instead of ELU
    - Second conv in each block replaced by depthwise-separable conv
    All pooling sizes + overall architecture remain EXACTLY K2C2.
    """

    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_ch, out_ch, pool_kernel):
            return nn.Sequential(
                # ------- Conv 1 (normal) -------
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),

                # ------- Conv 2 (depthwise-separable) -------
                DepthwiseSeparableConv(out_ch),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),

                # dropout same as baseline k2c2
                nn.Dropout2d(0.2),

                # ------- exact original pooling schedule -------
                nn.MaxPool2d(pool_kernel)
            )

        # 5 blocks (same as K2C2)
        self.block1 = conv_block(1,   32, (2, 4))
        self.block2 = conv_block(32,  64, (2, 4))
        self.block3 = conv_block(64, 128, (2, 4))
        self.block4 = conv_block(128, 256, (3, 5))
        self.block5 = conv_block(256, 512, (4, 4))

        # global / classifier EXACTLY the same as K2C2
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.gap(x)
        x = self.classifier(x)
        return x.squeeze(-1).squeeze(-1)
