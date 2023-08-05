import numpy as np
import torch
import torchvision
from torch.utils.data import random_split

from ...complete_dataset import DatasetEncapsulator

default_MNIST_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(
        # (0.1307,), (0.3081,))
    ]
)


def logit(x, alpha=1e-6):
    x = x * (1 - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1 - x)


def to_01(
    x,
):
    return x * (255.0 / 256.0) + (torch.rand_like(x) / 256.0)


def to_binary(
    x,
):
    return (x > 0.5).float()


transform_logit = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        to_01,
        logit,
    ]
)
transform_logit_padded32 = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Pad(2),
        to_01,
        logit,
    ]
)


transform_binary = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        to_binary,
    ]
)


class MNIST:
    def __init__(
        self,
        root_dir: str,
        transform=default_MNIST_transform,
        target_transform=None,
        download: bool = False,
        seed=None,
        input_size=(1, 28, 28),
        **kwargs,
    ):
        self.mnist_train = torchvision.datasets.MNIST(
            root=root_dir, train=True, download=download, transform=transform
        )
        self.mnist_test = torchvision.datasets.MNIST(
            root=root_dir, train=False, download=download, transform=transform
        )

        self.input_size = input_size

        self.dataset_test = DatasetEncapsulator(
            input_size=self.input_size,
            dataset=self.mnist_test,
        )
        self.dataset_train, self.dataset_val = random_split(
            self.mnist_train, [0.8, 0.2]
        )
        self.dataset_train = DatasetEncapsulator(
            input_size=self.input_size,
            dataset=self.dataset_train,
        )
        self.dataset_val = DatasetEncapsulator(
            input_size=self.input_size,
            dataset=self.dataset_val,
        )

    def get_dim_input(
        self,
    ):
        return (1, 28, 28)

    def get_dim_output(
        self,
    ):
        return 10


class MNISTLogitTransformed(MNIST):
    def __init__(
        self,
        root_dir: str,
        transform=transform_logit,
        target_transform=None,
        download: bool = False,
        seed=None,
        **kwargs,
    ):
        super().__init__(
            root_dir, transform, target_transform, download, seed, **kwargs
        )

    def transform_back(self, x):
        return torch.sigmoid(x)


class MNISTLogitTransformedPadded32(MNIST):
    def __init__(
        self,
        root_dir: str,
        transform=transform_logit_padded32,
        target_transform=None,
        download: bool = False,
        seed=None,
        input_size=(1, 32, 32),
        **kwargs,
    ):
        super().__init__(
            root_dir,
            transform,
            target_transform,
            download,
            seed,
            input_size=input_size,
            **kwargs,
        )

    def get_dim_input(
        self,
    ):
        return (1, 32, 32)

    def transform_back(self, x):
        return torch.sigmoid(x)


class BinaryMNIST(MNIST):
    def __init__(
        self,
        root_dir: str,
        transform=transform_binary,
        target_transform=None,
        download: bool = False,
        seed=None,
        **kwargs,
    ):
        super().__init__(
            root_dir, transform, target_transform, download, seed, **kwargs
        )

    def transform_back(self, x):
        return x
