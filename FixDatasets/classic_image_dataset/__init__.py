from .CIFAR10 import CIFAR10
from .CIFAR100 import CIFAR100
from .MNIST import MNIST, MNISTLogitTransformed, BinaryMNIST
from .FashionMNIST import FashionMNIST, FashionMNISTLogitTransformed, BinaryFashionMNIST
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
}