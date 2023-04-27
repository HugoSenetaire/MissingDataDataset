# Need to make an augmented dataset for the training here ?
import torch
from torch.utils.data import Dataset

class CompleteDatasets():
    def __init__(self, dataset_train, dataset_test, dataset_val, dim_input, dim_output, parameters = None, transform_back = None,):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.parameters = parameters
        if transform_back is None:
            self.transform_back = lambda x: x
        else :
            self.transform_back = transform_back

    def get_dim_input(self,):
        return self.dim_input

    def get_dim_output(self,):
        return self.dim_output
    


