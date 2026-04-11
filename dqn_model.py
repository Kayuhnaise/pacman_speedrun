import torch
import torch.nn as nn


class DQNCNN(nn.Module):
    def __init__(self, input_channels: int, rows: int, cols: int, num_actions: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, rows, cols)
            n_flat = self.features(dummy).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)