from .categorical import Categorical
from .ising import Ising
from .poisson_ds import Poisson

dic_discrete_dataset = {"categorical": Categorical, "poisson": Poisson, "ising": Ising}


def get_discrete_dataset(args_dict):
    dataset_name = args_dict["dataset_name"]
    if dataset_name in dic_discrete_dataset.keys():
        current_dataset = dic_discrete_dataset[dataset_name]
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    complete_dataset = current_dataset(**args_dict["dataset_params"])
    return complete_dataset
