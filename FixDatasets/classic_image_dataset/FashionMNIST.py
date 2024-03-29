
import torchvision 
import numpy as np
import torch
from torch.utils.data import TensorDataset
from ...complete_dataset import DatasetEncapsulator
from torch.utils.data import random_split

default_MNIST_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    # torchvision.transforms.Normalize(
                                        # (0.1307,), (0.3081,))
                                    ])

def logit(x, alpha=1e-6):
    x = x * (1 - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1 - x)

def to_01(x, ):
    return x * (255. / 256.) + (torch.rand_like(x) / 256.)

def to_binary(x, ):
    return (x > 0.5).float()


transform_logit = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),  
                                    to_01,
                                    logit,])

transform_binary = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    to_binary,])



class FashionMNIST():
    def __init__(self,
            root_dir: str,
            transform = default_MNIST_transform,
            target_transform = None,
            download: bool = False,
            seed = None,
            **kwargs,):

        self.fashion_mnist_train = torchvision.datasets.FashionMNIST(root = root_dir, train=True, download=download, transform=transform)
        self.fashion_mnist_test  = torchvision.datasets.FashionMNIST(root = root_dir, train=False, download=download, transform=transform)

        self.input_size = (1,28,28)

        self.dataset_test = DatasetEncapsulator(input_size = self.input_size, dataset = self.fashion_mnist_test,)
        self.dataset_train, self.dataset_val = random_split(self.fashion_mnist_train, [0.8, 0.2])
        self.dataset_train = DatasetEncapsulator(input_size = self.input_size, dataset = self.dataset_train,)
        self.dataset_val = DatasetEncapsulator(input_size = self.input_size, dataset = self.dataset_val,)


    def get_dim_input(self,):
        return (1,28,28)

    def get_dim_output(self,):
        return 10

class FashionMNISTLogitTransformed(FashionMNIST):
    def __init__(self,
            root_dir: str,
            transform = transform_logit,
            target_transform = None,
            download: bool = False,
            seed = None,
            **kwargs,):

        super().__init__(root_dir, transform, target_transform, download, seed, **kwargs)

    def transform_back(self, x):
        return torch.sigmoid(x)


class BinaryFashionMNIST(FashionMNIST):
    def __init__(self,
            root_dir: str,
            transform = transform_binary,
            target_transform = None,
            download: bool = False,
            seed = None,
            **kwargs,):

        super().__init__(root_dir, transform, target_transform, download, seed, **kwargs)

    def transform_back(self, x):
        return x
    


