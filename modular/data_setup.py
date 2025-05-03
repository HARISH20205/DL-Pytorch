import torch
import torch.nn as nn

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

print(torch.__version__, torch.cuda.is_available())


def create_dataloaders(train_dir, test_dir, transform, batch_size=16, num_workers=4):
    train_data = datasets.ImageFolder(
        root=train_dir, transform=transform, target_transform=None
    )
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names
