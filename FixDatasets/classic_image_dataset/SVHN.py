
from random import random
import torchvision 
import numpy as np
import torch
from ...complete_dataset import DatasetEncapsulator
from torch.utils.data import random_split
default_SVHN_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    ])


class SVHN():
    def __init__(self,
            root_dir: str,
            transform = default_SVHN_transform,
            target_transform = None,
            download: bool = False,
            seed = None,
            **kwargs,):

        self.SVHN_train = torchvision.datasets.SVHN(root = root_dir, split = 'train', download=download, transform=transform)
        self.SVHN_test  = torchvision.datasets.SVHN(root = root_dir, split = 'test', download=download, transform=transform)

        self.input_size = (3,32,32)

        self.dataset_test = DatasetEncapsulator(input_size = self.input_size, dataset = self.SVHN_test,)
        self.dataset_train, self.dataset_val = random_split(self.SVHN_train, [0.8, 0.2])
        self.dataset_train = DatasetEncapsulator(input_size = self.input_size, dataset = self.dataset_train,)
        self.dataset_val = DatasetEncapsulator(input_size = self.input_size, dataset = self.dataset_val,)

    def get_dim_input(self,):
        return self.input_size

    def get_dim_output(self,):
        return 10

    


