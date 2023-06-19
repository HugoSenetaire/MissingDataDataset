import argparse
from .FreeDatasets import FREE_DATASETS
from .FixDatasets import fix_dataset_dict
import os

list_dataset = list(fix_dataset_dict.keys()) + FREE_DATASETS

def default_args_missingdatadataset(parser = None,root_default = None ):
    if parser is None :
        parser = argparse.ArgumentParser()


    # DATASET AND MASKING :
        ## Dataset
    parser.add_argument('--download', type=bool, default=True, help='Download the dataset')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        choices=list_dataset,
                        help='The name of dataset')
    parser.add_argument('--dataset_parameters', type=dict, default={},
                        help='Parameters for the dataset')
    parser.add_argument('--static_generator_name', type=str, default=None,)
    parser.add_argument('--static_generator_parameters', type=dict, default={},)
    parser.add_argument('--dynamic_generator_name', type=str, default=None,)
    parser.add_argument('--dynamic_generator_parameters', type=dict, default={},)

    parser.add_argument('--yamldataset', type=str, default=None, help='YAML File path to override the dataset parameters')
    parser.add_argument('--seed', type=int, default=12, help='Seed for the random number generator')
    
    if root_default is None :
        raise ValueError("root_default should be provided")
    parser.add_argument('--root', type=str, default= root_default,)

    return parser