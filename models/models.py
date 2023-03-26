from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_classes: int):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return self.softmax(x)


class CNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
