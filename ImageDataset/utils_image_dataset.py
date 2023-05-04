
from .MNISTDataset import MnistDataset, BinaryMnistDataset, MnistDatasetLogitTransformed
from .FashionMNIST import FashionMNIST, BinaryFashionMnistDataset, FashionMNISTLogitTransformed
from .CIFAR100 import CIFAR100
from .CIFAR10 import CIFAR10
from .UTKFaceDataset import UTKFace
from .d1regression_1 import d1Regression_1
from .d1regression_2 import d1Regression_2
from .cell_count import CellCount
from .steering_angle import SteeringAngle
from.head_pose_biwi import HeadPoseBIWI
dic_image_dataset = {
    "MNIST" : MnistDataset,
    "MNISTlogit" : MnistDatasetLogitTransformed,
    "FashionMNIST": FashionMNIST,
    "FashionMNISTlogit": FashionMNISTLogitTransformed,
    "CIFAR100" : CIFAR100,
    "CIFAR10" : CIFAR10,
    "BinaryMNIST" : BinaryMnistDataset,
    "BinaryFashionMNIST" : BinaryFashionMnistDataset,
    "UTKFace" : UTKFace,
    "1d_regression_1" : d1Regression_1,
    "1d_regression_2" : d1Regression_2,
    'CellCount' : CellCount,
    'SteeringAngle' : SteeringAngle,
    'HeadPoseBIWI' : HeadPoseBIWI,
}

def get_image_dataset(args_dict):
    dataset_name = args_dict["dataset_name"]
    if dataset_name in dic_image_dataset.keys():
        current_dataset = dic_image_dataset[dataset_name] 
    else :
        raise ValueError(f"Dataset {dataset_name} not found")

    
    complete_dataset = current_dataset(root_dir = args_dict["root"], seed=args_dict["seed"], download = args_dict["download"],)

    return complete_dataset