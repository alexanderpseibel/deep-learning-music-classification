#src/models/cnn_generic.py
import torch
import torch.nn as nn


class GenericCNN(nn.Module):
    """
    A fully configurable CNN.

    Configurable parameters:
    - channels: list[int]                   e.g. [32, 64, 128, 256, 512]
    - kernels:  list[list[int,int]]         e.g. [[3,3],[3,3],[5,5],[3,3],[3,3]]
    - pooling:  list[list[int,int]]         e.g. [[2,4],[2,4],[3,5],[4,4],[4,4]]
    - dropout: float                        e.g. 0.2
    - activation: "elu" or "relu"
    """

    def __init__(
        self,
        num_classes,
        channels,
        kernels,
        pooling,
        dropout=0.2,
        activation="elu"
    ):
        super().__init__()

        assert len(channels) == len(kernels) == len(pooling), \
            "channels, kernels, pooling must have same length (#blocks)"

        self.num_blocks = len(channels)
        self.dropout = dropout
        self.activation_name = activation.lower()

        # choose activation
        def act():
            return nn.ELU() if self.activation_name == "elu" else nn.ReLU()

        blocks = []
        in_ch = 1

        # Build each block
        for out_ch, k, pool in zip(channels, kernels, pooling):
            k_t, k_f = k
            p_t, p_f = pool

            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(k_t, k_f),
                          padding=(k_t // 2, k_f // 2)),
                nn.BatchNorm2d(out_ch),
                act(),

                nn.Conv2d(out_ch, out_ch, kernel_size=(k_t, k_f),
                          padding=(k_t // 2, k_f // 2)),
                nn.BatchNorm2d(out_ch),
                act(),

                nn.Dropout2d(self.dropout),
                nn.MaxPool2d((p_t, p_f))
            )

            blocks.append(block)
            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)

        # GAP + classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.gap(x)             # (B, C, 1, 1)
        x = self.classifier(x)      # (B, num_classes, 1, 1)
        return x.squeeze(-1).squeeze(-1)
