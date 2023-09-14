from .CIFAR10 import CIFAR10
from .CIFAR100 import CIFAR100
from .FashionMNIST import BinaryFashionMNIST, FashionMNIST, FashionMNISTLogitTransformed
from .MNIST import (
    MNIST,
    BinaryMNIST,
    MNISTLogitTransformed,
    MNISTLogitTransformedPadded32,
    MNISTPadded32
)
from .SVHN import SVHN

CLASSIC_IMAGE_DATASETS = {
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "MNIST": MNIST,
    "MNISTLogitTransformed": MNISTLogitTransformed,
    "BinaryMNIST": BinaryMNIST,
    "FashionMNIST": FashionMNIST,
    "FashionMNISTLogitTransformed": FashionMNISTLogitTransformed,
    "BinaryFashionMNIST": BinaryFashionMNIST,
    "SVHN": SVHN,
    "MNISTLogitTransformedPadded32": MNISTLogitTransformedPadded32,
    "MNISTPadded32": MNISTPadded32,
}
