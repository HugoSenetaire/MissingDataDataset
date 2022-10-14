
import torchvision 
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

default_CIFAR100_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    ])


class CIFAR100():
    def __init__(self,
            root_dir: str,
            transform = default_CIFAR100_transform,
            target_transform = None,
            download: bool = False,
            seed = None,
            **kwargs,):

        self.CIFAR100_train = torchvision.datasets.CIFAR100(root = root_dir, train=True, download=download, transform=transform)
        self.CIFAR100_test  = torchvision.datasets.CIFAR100(root = root_dir, train=False, download=download, transform=transform)

        self.data_train = torch.stack([self.CIFAR100_train.__getitem__(i)[0] for i in range(len(self.CIFAR100_train))])
        self.data_test = torch.stack([self.CIFAR100_test.__getitem__(i)[0] for i in range(len(self.CIFAR100_test))])
        self.target_train = self.CIFAR100_train.targets
        self.target_test = self.CIFAR100_test.targets

        self.data_train = self.data_train.reshape(-1,1,32,32)
        index_train, index_val = train_test_split(np.range(len(self.data_train)), seed= seed)

        self.data_test = self.data_test.reshape(-1,1,32,32)
        self.dataset_train = TensorDataset(self.data_train[index_train], self.target_train[index_train])
        self.dataset_val = TensorDataset(self.data_train[index_val], self.target_train[index_val])
        self.dataset_test = TensorDataset(self.data_test, self.target_test)

    def get_dim_input(self,):
        return (1,32,32)

    def get_dim_output(self,):
        return 100

    


