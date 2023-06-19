import os
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
import torch

class CustomImageDataset(Dataset):
    def __init__(self, image_np, target_np, transform=None, target_transform=None):
        self.image_np = image_np
        self.target_np = target_np 
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.target_np)

    def __getitem__(self, idx):
        image = self.image_np[idx]
        label = torch.tensor(self.target_np[idx],)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label