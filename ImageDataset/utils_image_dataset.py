
from .MNISTDataset import MnistDataset, BinaryMnistDataset
from .CIFAR100 import CIFAR100
from .CIFAR10 import CIFAR10


dic_image_dataset = {
    "MNIST" : MnistDataset,
    "CIFAR100" : CIFAR100,
    "CIFAR10" : CIFAR10,
    "BinaryMNIST" : BinaryMnistDataset,
}

def get_image_dataset(args_dict):
    dataset_name = args_dict["dataset_name"]
    if dataset_name in dic_image_dataset.keys():
        current_dataset = dic_image_dataset[dataset_name] 
    else :
        raise ValueError(f"Dataset {dataset_name} not found")

    
    complete_dataset = current_dataset(root_dir = args_dict["root"], seed=args_dict["seed"], download = args_dict["download"],)

    return complete_dataset