# Need to make an augmented dataset for the training here ?
import torch
from torch.utils.data import Dataset

class CompleteDatasets():
    def __init__(self, dataset_train, dataset_test, dataset_val, dim_input, dim_output):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val
        self.dim_input = dim_input
        self.dim_output = dim_output
        

    def get_dim_input(self,):
        return self.dim_input

    def get_dim_output(self,):
        return self.dim_output



