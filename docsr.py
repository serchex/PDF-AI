import torch
import torch.nn as nn

class DocSR(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(1, 64, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(18):
            layers.append(nn.Conv2d(64, 64, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) + x  # residual learning
