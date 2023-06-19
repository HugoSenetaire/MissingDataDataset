
from random import random
import torchvision 
from torch.utils.data import random_split
from ...complete_dataset import DatasetEncapsulator
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
      
        self.input_size = (3,32,32)

        self.dataset_test = DatasetEncapsulator(input_size = self.input_size, dataset = self.CIFAR10_test,)
        self.dataset_train, self.dataset_val = random_split(self.CIFAR10_train, [0.8, 0.2])
        self.dataset_train = DatasetEncapsulator(input_size = self.input_size, dataset = self.dataset_train,)
        self.dataset_val = DatasetEncapsulator(input_size = self.input_size, dataset = self.dataset_val,)

    def get_dim_input(self,):
        return self.input_size

    def get_dim_output(self,):
        return 10
    
    def transform_back(self, x):
        return x

    


