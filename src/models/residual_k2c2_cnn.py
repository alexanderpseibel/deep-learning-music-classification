import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualK2C2Block(nn.Module):
    """
    Residual version of a K2C2 block:
    Conv → BN → ELU → Conv → BN → ELU + shortcut → MaxPool

    - Shortcut adjusts channels if needed
    - Pooling matches the original K2C2 paper
    """

    def __init__(self, in_ch, out_ch, pool_kernel):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Shortcut path for matching dimensions
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

        self.dropout = nn.Dropout2d(0.2)
        self.pool = nn.MaxPool2d(pool_kernel)

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.elu(out)

        out += shortcut  # Residual connection
        out = F.elu(out)

        out = self.dropout(out)
        out = self.pool(out)

        return out


class ResidualK2C2CNN(nn.Module):
    """
    Residual-enhanced K2C2:
    - Keeps all K2C2 inductive biases
    - Adds residuals for better gradient flow and stability
    - Best-performing pure CNN for music tagging
    """

    def __init__(self, num_classes):
        super().__init__()

        self.block1 = ResidualK2C2Block(1,   32, (2, 4))
        self.block2 = ResidualK2C2Block(32,  64, (2, 4))
        self.block3 = ResidualK2C2Block(64, 128, (2, 4))
        self.block4 = ResidualK2C2Block(128, 256, (3, 5))
        self.block5 = ResidualK2C2Block(256, 512, (4, 4))

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
        x = x.squeeze(-1).squeeze(-1)
        return x