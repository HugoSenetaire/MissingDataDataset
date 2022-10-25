import argparse
from .ImageDataset import *
from .TabularDataset import *
import os

list_dataset = DATASETS_TENSOR + list(dic_image_dataset.keys())

def default_args_missingdatadataset(parser = None,root_default = None ):
    if parser is None :
        parser = argparse.ArgumentParser()


    # DATASET AND MASKING :
        ## Dataset
    parser.add_argument('--download', type=bool, default=True, help='Download the dataset')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        choices=list_dataset,
                        help='The name of dataset')
    parser.add_argument('--dim', type = int, default = 10, help='Dimension of the dataset for some tabular artificial data')
        ## Masking
    parser.add_argument('--missing_mechanism', type=str, default="mcar",
                        help='Choose between different type of missing mechanism')
    parser.add_argument('--p_obs', type=float, default=0.0,
                        help= 'Proportion of features with no missing values, Used in MCAR and MAR')
    parser.add_argument('--p_missing', type=float, default=0.85,
                        help = 'Proportion of missing values to generate for variables which will have missing values. Used in Mcar Mar Mnar')
    parser.add_argument('--p_params', type=float, default=0.6,
                        help = 'Proportion of variables that will be used for the logistic masking model in mnar_logistic')
    parser.add_argument('--exclude_inputs', action='store_true',
                        help = 'Exclude non masked data from having influence on the masking model in MNAR Logistic')
    parser.add_argument("-q", '--quantile', type=float, default=.5,
                        help='distance quantile to select epsilon')
    parser.add_argument('--cut', type=str, choices=['both', 'upper', 'lower'], default='both',
                        help = "Used for mnar_quantiles Where the cut should be applied. For instance, if q=0.25 and cut='upper', \
                        then missing values will be generated in the upper quartiles of selected variables.")
    
    if root_default is None :
        raise ValueError("root_default should be provided")
    parser.add_argument('--root', type=str, default= root_default,)

    return parser