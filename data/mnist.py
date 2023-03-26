from typing import Optional
import numpy as np
import torch
from torchvision import datasets, transforms


class MNISTDataset(datasets.MNIST):

    N_CLASSES = 10

    def __init__(self, root: str, train: bool):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        super().__init__(root=root, train=train, download=True, transform=transform)

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]
        x = self.transform(x)

        return x, y
