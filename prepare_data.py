from .DatasetUtils import CompleteDatasets
from .DiscreteDataset import dic_discrete_dataset, get_discrete_dataset
from .ImageDataset import dic_image_dataset, get_image_dataset
from .MaskDataset import create_mask_dataset
from .open_yaml import update_config_from_paths
from .TabularDataset import DATASETS_TENSOR, get_dataset_from_tensor
from .DatasetFromMAF import dic_maf_dataset, get_maf_dataset

def get_vanilla_dataset(
    args_dict,
):
    if args_dict["dataset_name"] in DATASETS_TENSOR:
        (
            dataset_train,
            dataset_test,
            dataset_val,
            dim_input,
            dim_output,
            parameters,
        ) = get_dataset_from_tensor(
            args_dict,
        )
        complete_dataset = CompleteDatasets(
            dataset_train,
            dataset_test,
            dataset_val,
            dim_input,
            dim_output,
            parameters=parameters,
        )
    elif args_dict["dataset_name"] in dic_image_dataset:
        complete_dataset = get_image_dataset(
            args_dict,
        )
    elif args_dict["dataset_name"] in dic_discrete_dataset:
        complete_dataset = get_discrete_dataset(
            args_dict,
        )
    elif args_dict["dataset_name"] in dic_maf_dataset :
        complete_dataset = get_maf_dataset(
            name = args_dict['dataset_name'],
            root = args_dict["root"],
        )
    else:
        raise ValueError("Dataset {} not recognized".format(args_dict["dataset_name"]))

    return complete_dataset


def augment_dataset_with_mask(
    args_dict,
    complete_dataset,
):
    if args_dict["dataset_name"] not in dic_discrete_dataset:
        complete_masked_dataset = create_mask_dataset(
            args_dict,
            complete_dataset,
            DATASETS_TENSOR=DATASETS_TENSOR,
            dic_image_dataset=dic_image_dataset,
        )
    else:
        complete_masked_dataset = create_mask_dataset(
            args_dict,
            complete_dataset,
            DATASETS_TENSOR=DATASETS_TENSOR,
            dic_image_dataset=dic_image_dataset,
        )
    return complete_masked_dataset


def get_dataset(
    args_dict,
):
    # args_dict = update_config_from_paths(args_dict,)
    complete_dataset = get_vanilla_dataset(args_dict=args_dict)
    complete_masked_dataset = augment_dataset_with_mask(
        complete_dataset=complete_dataset, args_dict=args_dict
    )

    return complete_dataset, complete_masked_dataset


class InfiniteDataLoader(object):
    """docstring for InfiniteDataLoader"""

    def __init__(self, dataloader):
        super(InfiniteDataLoader, self).__init__()
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            # Reached end of the dataset
            self.data_iter = iter(self.dataloader)
            data = next(self.data_iter)

        return data

    def __len__(self):
        return len(self.dataloader)
