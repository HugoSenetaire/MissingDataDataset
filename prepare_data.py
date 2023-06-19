
from .MaskDataset import create_mask_dataset
from .FixDatasets import fix_dataset_dict, get_fix_dataset
from .FreeDatasets import FREE_DATASETS, get_free_datasets



def get_vanilla_dataset(
    args_dict,
):
    """
    Given a complete dataset with some masks, return a complete dataset without masks
    """
    if args_dict["dataset_name"] in fix_dataset_dict.keys():
        dataset = get_fix_dataset(args_dict)
    elif args_dict['dataset_name'] in FREE_DATASETS:
        dataset = get_free_datasets(args_dict)
    else:
        raise ValueError("Dataset {} not recognized".format(args_dict["dataset_name"]))
    
    return dataset

def get_dataset(
    args_dict,
):
    # args_dict = update_config_from_paths(args_dict,)
    complete_dataset = get_vanilla_dataset(args_dict=args_dict)
    complete_masked_dataset = create_mask_dataset(
        complete_dataset=complete_dataset, args_dict=args_dict
    )

    return complete_dataset, complete_masked_dataset



