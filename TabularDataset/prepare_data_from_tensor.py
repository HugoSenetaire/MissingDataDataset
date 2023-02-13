
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from .utilsUCI import tensor_dataset_loader
from ..MaskDataset.MaskAugmentedDataset import DatasetMaskAugmented
from torch.utils.data import TensorDataset
import torch
import numpy as np


def get_dataset_from_tensor(args_dict,):
    data_true, Y , parameters = tensor_dataset_loader(args_dict["dataset_name"], args=args_dict)
    if parameters is None :
        data_true = scale(data_true)
    data_true = torch.tensor(data_true, dtype=torch.float32).unsqueeze(1)
    try :
        if args_dict["problem_type"] == "classification" :
            Y = torch.tensor(Y, dtype=torch.int64)
            dim_output = len(np.unique(Y))
        elif args_dict["problem_type"] == "regression" or args_dict["problem_type"] == "likelihood":
            dim_output = 1
            Y = torch.tensor(Y, dtype=torch.float32)
        elif args_dict["problem_type"] == "not_needed":
            dim_output = 1
        else :
            raise ValueError("Error, the problem type is not recognized")
    except KeyError:
        Y = torch.tensor(Y, dtype=torch.float32)
        dim_output = None

    dim_input = data_true[0].shape


    data_true = torch.tensor(data_true, dtype=torch.float32)
    data_train, data_val, target_train, mask_val = train_test_split(data_true, Y, test_size=0.1, random_state=args_dict["seed"])
    data_train, data_test, target_train, mask_test = train_test_split(data_train, target_train, test_size=0.2, random_state=args_dict["seed"])

    
    dataset_train = TensorDataset(data_train, target_train)
    dataset_val = TensorDataset(data_val, mask_val)
    dataset_test = TensorDataset(data_test, mask_test)

    return dataset_train, dataset_test, dataset_val, dim_input, dim_output, parameters
