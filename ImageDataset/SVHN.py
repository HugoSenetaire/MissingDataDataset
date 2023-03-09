
from random import random
import torchvision 
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

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

        self.data_train = torch.stack([self.SVHN_train.__getitem__(i)[0] for i in range(len(self.SVHN_train))])
        self.data_test = torch.stack([self.SVHN_test.__getitem__(i)[0] for i in range(len(self.SVHN_test))])
        self.target_train = torch.stack([torch.tensor(self.SVHN_train.__getitem__(i)[1], dtype=torch.long) for i in range(len(self.SVHN_train))])
        self.target_test = torch.stack([torch.tensor(self.SVHN_test.__getitem__(i)[1], dtype=torch.long) for i in range(len(self.SVHN_test))])

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

    


