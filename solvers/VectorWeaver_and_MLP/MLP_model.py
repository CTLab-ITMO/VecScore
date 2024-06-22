import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, D_in, D_out):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(D_in, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, D_out)
        )

    def forward(self, x):
        return self.model(x)
