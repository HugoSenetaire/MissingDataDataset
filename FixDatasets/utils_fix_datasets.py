
from .classic_image_dataset import CLASSIC_IMAGE_DATASETS
from .ebm_for_regression_dataset import EBM_FOR_REGRESSION_DATASETS
from .discrete_datasets import dic_discrete_dataset
from .dataset_from_maf import dic_maf_dataset, get_maf_dataset


liste_datasets = [CLASSIC_IMAGE_DATASETS, EBM_FOR_REGRESSION_DATASETS, dic_discrete_dataset, dic_maf_dataset,]

# Check that all datasets have different names
for pairs in [(dic1, dic2) for dic1 in liste_datasets for dic2 in liste_datasets]:
    if pairs[0] is not pairs[1]:
        assert not any([key in pairs[0] for key in pairs[1]]), "Two datasets have the same name"

fix_dataset_dict = {**CLASSIC_IMAGE_DATASETS, **EBM_FOR_REGRESSION_DATASETS, **dic_discrete_dataset, **dic_maf_dataset,}

def get_fix_dataset(args_dict):
    dataset_name = args_dict["dataset_name"]
    current_dataset = fix_dataset_dict[dataset_name]
    complete_dataset = current_dataset(root_dir = args_dict["root"], **args_dict["dataset_parameters"])
    return complete_dataset