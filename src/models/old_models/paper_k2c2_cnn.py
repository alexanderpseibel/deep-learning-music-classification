#src/models/old_models/paper_k2c2_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperK2C2CNN(nn.Module):
    """
    Paper-exact K2C2 CNN (Choi et al., 2017),
    adapted to 128×3000 input but otherwise identical.
    """

    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_ch, out_ch, pool_kernel):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ELU(),

                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ELU(),

                nn.MaxPool2d(pool_kernel)
            )

        self.block1 = conv_block(1,   64, (2, 4))
        self.block2 = conv_block(64, 128, (2, 4))
        self.block3 = conv_block(128, 128, (2, 4))
        self.block4 = conv_block(128, 256, (3, 5))
        self.block5 = conv_block(256, 256, (4, 4))

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.gap(x)
        x = self.classifier(x)
        return x.squeeze(-1).squeeze(-1)
