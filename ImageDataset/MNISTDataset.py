
import torchvision 
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

default_MNIST_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    # torchvision.transforms.Normalize(
                                    #     (0.1307,), (0.3081,))
                                    ])


class MnistDataset():
    def __init__(self,
            root_dir: str,
            transform = default_MNIST_transform,
            target_transform = None,
            download: bool = False,
            seed = None,
            **kwargs,):

        self.mnist_train = torchvision.datasets.MNIST(root = root_dir, train=True, download=download, transform=transform)
        self.mnist_test  = torchvision.datasets.MNIST(root = root_dir, train=False, download=download, transform=transform)

        self.data_train = torch.stack([self.mnist_train.__getitem__(i)[0] for i in range(len(self.mnist_train))])
        self.data_test = torch.stack([self.mnist_test.__getitem__(i)[0] for i in range(len(self.mnist_test))])
        self.target_train = self.mnist_train.targets
        self.target_test = self.mnist_test.targets

        self.data_train = self.data_train.reshape(-1,1,28,28)
        index_train, index_val = train_test_split(np.arange(len(self.data_train)), random_state= seed)

        self.data_test = self.data_test.reshape(-1,1,28,28)
        self.dataset_train = TensorDataset(self.data_train[index_train], self.target_train[index_train])
        self.dataset_val = TensorDataset(self.data_train[index_val], self.target_train[index_val])
        self.dataset_test = TensorDataset(self.data_test, self.target_test)

    def get_dim_input(self,):
        return (1,28,28)

    def get_dim_output(self,):
        return 10

    


