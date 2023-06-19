import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from .UCI_datasets.utilsUCI import uci_dataset_loader, UCI_DATASETS
from .ArtificialDatasets import generated_dataset_loader, GENERATED_DATASETS
from ..complete_dataset import CompleteDatasets, DictTensorDataset

FREE_DATASETS = GENERATED_DATASETS + UCI_DATASETS



    
def get_free_datasets(
    args_dict,
):
    '''
    Loading data from tensor based (ie, the separation between train, test and validation is yet to be done)
    For instance, UCI datasets, artificial datasets, etc.
    '''
    if args_dict["dataset_name"] in GENERATED_DATASETS:
        data_true, Y, parameters = generated_dataset_loader(args_dict["dataset_name"], args=args_dict["dataset_parameters"])
    elif args_dict["dataset_name"] in UCI_DATASETS:
        data_true, Y, parameters = uci_dataset_loader(args_dict["dataset_name"], args=args_dict["dataset_parameters"])
    
    data_true = torch.tensor(data_true, dtype=torch.float32).unsqueeze(1)

    try:
        if args_dict["problem_type"] == "classification": # TODO: I think I should actually put that directly in the dataset.
            Y = torch.tensor(Y, dtype=torch.int64)
            dim_output = len(np.unique(Y))
        elif (
            args_dict["problem_type"] == "regression"
            or args_dict["problem_type"] == "likelihood"
        ):
            dim_output = 1
            Y = torch.tensor(Y, dtype=torch.float32)
        elif args_dict["problem_type"] == "not_needed":
            dim_output = 1
        else:
            raise ValueError("Error, the problem type is not recognized")
    except KeyError:
        Y = torch.tensor(Y, dtype=torch.float32)
        dim_output = None

    dim_input = data_true[0].shape
    data_train, data_val, target_train, target_val = train_test_split(
        data_true, Y, test_size=0.1, random_state=args_dict["seed"]
    )
    data_train, data_test, target_train, target_test = train_test_split(
        data_train, target_train, test_size=0.2, random_state=args_dict["seed"]
    )

    if parameters is None:
        data_train = scale(data_train)
        data_train = torch.tensor(data_train, dtype=torch.float32)
        data_val = scale(data_val)
        data_val = torch.tensor(data_val, dtype=torch.float32)
        data_test = scale(data_test)
        data_test = torch.tensor(data_test, dtype=torch.float32)

    dataset_train = DictTensorDataset(dim_input, data_train, target_train)
    dataset_val = DictTensorDataset(dim_input, data_val, target_val)
    dataset_test = DictTensorDataset(dim_input, data_test, target_test)

    complete_dataset = CompleteDatasets(dataset_train=dataset_train,
                                        dataset_val=dataset_val,
                                        dataset_test=dataset_test,
                                        dim_input=dim_input,
                                        dim_output=dim_output,
                                        parameters=parameters)
    

    return complete_dataset
