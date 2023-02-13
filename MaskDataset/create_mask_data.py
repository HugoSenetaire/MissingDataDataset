from .MaskAugmentedDataset import DatasetMaskAugmented
from .mask_utils_image import mask_loader_image
from .mask_utils_tabular import mask_loader_tabular
from ..DatasetUtils import CompleteDatasets

import torch
import numpy as np

def create_mask_dataset(args_dict, complete_dataset, DATASETS_TENSOR, dic_image_dataset,):
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
        mask_X, mask_Y, patterns = create_mask(X, Y, args=args_dict, seed=args_dict["seed"])
        mask_X = 1-mask_X.to(torch.float32)
        list_datasets.append(DatasetMaskAugmented(dataset, mask_X,))

    if parameters is not None :
        parameters["patterns"] = patterns
    complete_masked_dataset = CompleteDatasets(list_datasets[0],
                                            list_datasets[1],
                                            list_datasets[2],
                                            complete_dataset.get_dim_input(),
                                            complete_dataset.get_dim_output(),
                                            parameters=parameters)
    return complete_masked_dataset




def create_dataset_without_mask(complete_dataset,):
    """
    Given a complete dataset with some masks, return a complete dataset without masks
    """
    dataset_train = complete_dataset.dataset_train
    masks = dataset_train.mask.flatten(1)
    assert torch.any(masks.sum(-1) == masks.shape[1]) # Making sure that there is at least one data point without any mask

    data = []
    target = []
    mask = []

    for k in range(len(dataset_train)):
        current_mask = dataset_train[k]['mask']
        if current_mask.sum() == np.prod(current_mask.shape):
            data.append(dataset_train[k]['data'])
            target.append(dataset_train[k]['target'])
            mask.append(dataset_train[k]['mask'])

    new_dataset_train = torch.utils.data.TensorDataset(torch.stack(data), torch.stack(target),)
    new_dataset_train_mask_augmented = DatasetMaskAugmented(new_dataset_train, torch.stack(mask),)

    new_complete_dataset = CompleteDatasets(dataset_train = new_dataset_train_mask_augmented,
                                            dataset_test = complete_dataset.dataset_test,
                                            dataset_val = complete_dataset.dataset_val,
                                            dim_input = complete_dataset.get_dim_input(),
                                            dim_output = complete_dataset.get_dim_output(),
                                            parameters = complete_dataset.parameters)
    return new_complete_dataset