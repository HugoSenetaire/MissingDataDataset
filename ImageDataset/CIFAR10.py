
from random import random
import torchvision 
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

default_CIFAR10_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    ])


class CIFAR10():
    def __init__(self,
            root_dir: str,
            transform = default_CIFAR10_transform,
            target_transform = None,
            download: bool = False,
            seed = None,
            **kwargs,):

        self.CIFAR10_train = torchvision.datasets.CIFAR10(root = root_dir, train=True, download=download, transform=transform)
        self.CIFAR10_test  = torchvision.datasets.CIFAR10(root = root_dir, train=False, download=download, transform=transform)

        self.data_train = torch.stack([self.CIFAR10_train.__getitem__(i)[0] for i in range(len(self.CIFAR10_train))])
        self.data_test = torch.stack([self.CIFAR10_test.__getitem__(i)[0] for i in range(len(self.CIFAR10_test))])
        self.target_train = torch.tensor(self.CIFAR10_train.targets, dtype=torch.long)
        self.target_test = torch.tensor(self.CIFAR10_test.targets, dtype=torch.long)

        self.data_train = self.data_train.reshape(-1,3,32,32)
        index_train, index_val = train_test_split(np.arange(len(self.data_train)), random_state= seed)


        self.data_test = self.data_test.reshape(-1,3,32,32)
        self.dataset_train = TensorDataset(self.data_train[index_train], self.target_train[index_train])
        self.dataset_val = TensorDataset(self.data_train[index_val], self.target_train[index_val])
        self.dataset_test = TensorDataset(self.data_test, self.target_test)

    def get_dim_input(self,):
        return (3,32,32)

    def get_dim_output(self,):
        return 10

    


