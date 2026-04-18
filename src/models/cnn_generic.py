# src/models/cnn_generic.py
import torch.nn as nn


class GenericCNN(nn.Module):
    """
    Configurable CNN for mel spectrogram classification.

    Each block: Conv2d → BN → Act → Conv2d → BN → Act → Dropout2d → MaxPool2d
    Classifier: global average pool → 1×1 conv

    Args:
        channels: output channels per block, e.g. [32, 64, 128, 256, 512]
        kernels:  (time, freq) kernel size per block, e.g. [[3,3], ...]
        pooling:  (time, freq) pool size per block, e.g. [[2,4], ...]
        dropout:  Dropout2d probability
        activation: "elu" or "relu"
    """

    def __init__(self, num_classes, channels, kernels, pooling, dropout=0.2, activation="elu"):
        super().__init__()

        assert len(channels) == len(kernels) == len(pooling), \
            "channels, kernels, pooling must have the same length"

        self.activation_name = activation.lower()
        self.dropout = dropout

        def act():
            return nn.ELU() if self.activation_name == "elu" else nn.ReLU()

        in_ch = 1
        blocks = []
        for out_ch, k, pool in zip(channels, kernels, pooling):
            k_t, k_f = k
            p_t, p_f = pool
            blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, (k_t, k_f), padding=(k_t // 2, k_f // 2)),
                nn.BatchNorm2d(out_ch),
                act(),
                nn.Conv2d(out_ch, out_ch, (k_t, k_f), padding=(k_t // 2, k_f // 2)),
                nn.BatchNorm2d(out_ch),
                act(),
                nn.Dropout2d(self.dropout),
                nn.MaxPool2d((p_t, p_f)),
            ))
            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x.squeeze(-1).squeeze(-1)
