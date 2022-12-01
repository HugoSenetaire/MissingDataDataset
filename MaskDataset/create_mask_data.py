from .MaskAugmentedDataset import DatasetMaskAugmented
from .mask_utils_image import mask_loader_image
from .mask_utils_tabular import mask_loader_tabular
from ..DatasetUtils import CompleteDatasets

import torch

def create_mask_dataset(args_dict, complete_dataset,DATASETS_TENSOR, dic_image_dataset,):
    create_mask = None
    try :
        parameters = complete_dataset.parameters
    except AttributeError as e:
        print(e)
        parameters = None
    dataset_name = args_dict["dataset_name"]
    if dataset_name in DATASETS_TENSOR :
        create_mask = mask_loader_tabular
    elif dataset_name in dic_image_dataset :
        create_mask = mask_loader_image
    else :
        raise ValueError("Dataset {} not recognized".format(dataset_name))
    
    
    list_datasets= []
    for attr_name in ["dataset_train", "dataset_test", "dataset_val"]:
        dataset = getattr(complete_dataset, attr_name)
        try :
            X,Y = dataset.tensors[0], dataset.tensors[1]
        except KeyError:
            X = torch.stack([dataset.__getitem__(i)[0] for i in range(len(dataset))])
            Y = torch.stack([dataset.__getitem__(i)[1] for i in range(len(dataset))])
        mask_X, mask_Y = create_mask(X, Y, args=args_dict, seed=args_dict["seed"])
        mask_X = 1-mask_X.to(torch.float32)
        list_datasets.append(DatasetMaskAugmented(dataset, mask_X))

    complete_masked_dataset = CompleteDatasets(list_datasets[0],
                                            list_datasets[1],
                                            list_datasets[2],
                                            complete_dataset.get_dim_input(),
                                            complete_dataset.get_dim_output(),
                                            parameters=parameters)
    return complete_masked_dataset