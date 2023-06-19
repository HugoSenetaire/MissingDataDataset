import numpy as np
import torch
import tqdm

from torch.utils.data import ConcatDataset

from .mask_generators import dic_mask_generators, NoneMaskGenerator, dic_stats_generator
from .mask_augmented_dataset import DatasetMaskAugmented
from ..complete_dataset import CompleteDatasets


def create_mask_dataset(
    args_dict,
    complete_dataset,
):
    """
    Given a complete dataset, return a complete dataset with DatasetMaskAugmented, using generators to create 
    fixed and dynamic masks.
    """
    complete_dataset_cat = ConcatDataset([complete_dataset.dataset_train, complete_dataset.dataset_val, complete_dataset.dataset_test, ])
    if 'static_generator_name' in args_dict.keys() and args_dict['static_generator_name'] is not None :
        static_generator= dic_mask_generators[args_dict['static_generator_name']](**args_dict['static_generator_parameters'])
        if args_dict['static_generator_name'] in dic_stats_generator :
            static_generator.calculate_stats(complete_dataset_cat)

        print("Generate static masks")
        batch_size = args_dict['batch_size']
        masks = []
        for i in range(0, len(complete_dataset_cat), batch_size):
            indexes = np.linspace(i, min(i+batch_size, len(complete_dataset_cat)-1), min(batch_size, len(complete_dataset_cat)-i), dtype=int)
            batch = complete_dataset_cat.__getitem__(indexes)
            if len(indexes) == 1 :
                batch['data'] = batch['data'].unsqueeze(0)
            masks.append(static_generator(batch))
        masks = torch.cat(masks, 0)
        masks_train = masks[:len(complete_dataset.dataset_train)]
        masks_val = masks[len(complete_dataset.dataset_train):len(complete_dataset.dataset_train)+len(complete_dataset.dataset_val)]
        masks_test = masks[len(complete_dataset.dataset_train)+len(complete_dataset.dataset_val):]
    else :
        static_generator = NoneMaskGenerator()
        masks_train = torch.ones(len(complete_dataset.dataset_train), *complete_dataset.dim_input)
        masks_val = torch.ones(len(complete_dataset.dataset_val), *complete_dataset.dim_input)
        masks_test = torch.ones(len(complete_dataset.dataset_test), *complete_dataset.dim_input)
    


    if 'dynamic_generator_name' in args_dict.keys() and args_dict['dynamic_generator_name'] is not None :
        dynamic_generator= dic_mask_generators[args_dict['dynamic_generator_name']](**args_dict['dynamic_generator_parameters'])
        if args_dict['static_generator_name'] in dic_stats_generator :
            static_generator.calculate_stats(complete_dataset)
    else :
        dynamic_generator = NoneMaskGenerator()

    dataset_train = DatasetMaskAugmented(complete_dataset.dataset_train, masks_train, static_generator, dynamic_generator)
    dataset_val = DatasetMaskAugmented(complete_dataset.dataset_val, masks_val, static_generator, dynamic_generator)
    dataset_test = DatasetMaskAugmented(complete_dataset.dataset_test, masks_test, static_generator, dynamic_generator)

    if hasattr(complete_dataset, 'parameters') :
        param = complete_dataset.parameters
    else :
        param = None

    complete_dataset_masked = CompleteDatasets(dataset_train=dataset_train,
                                                dataset_test=dataset_test,
                                                dataset_val=dataset_val,
                                                dim_input=complete_dataset.dim_input,
                                                dim_output=complete_dataset.dim_output,
                                                parameters=param)

    
    return complete_dataset_masked


    


    

