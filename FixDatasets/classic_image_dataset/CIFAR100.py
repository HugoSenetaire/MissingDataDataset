
import torchvision 
import numpy as np
from ...complete_dataset import DatasetEncapsulator
from torch.utils.data import random_split

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

       
        self.input_size = (3,32,32)

        self.dataset_test = DatasetEncapsulator(input_size = self.input_size, dataset = self.CIFAR100_test,)
        self.dataset_train, self.dataset_val = random_split(self.CIFAR100_train, [0.8, 0.2])
        self.dataset_train = DatasetEncapsulator(input_size = self.input_size, dataset = self.dataset_train,)
        self.dataset_val = DatasetEncapsulator(input_size = self.input_size, dataset = self.dataset_val,)



    def get_dim_input(self,):
        return self.input_size

    def get_dim_output(self,):
        return 100

    


