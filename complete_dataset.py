# Need to make an augmented dataset for the training here ?
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

class DatasetEncapsulator(Dataset):
    '''
    Normalize the sample of the dataset to be a dictionnary with keys "data" and "target" with correct size

    Attributes
    ----------
    input_size : tuple (channel, size_1, size_2, ...)
        Dimension of the input, channel is mandatory.
    
    dataset : torch.utils.data.Dataset
        The dataset that will be encapsulated
    '''
    def __init__(self, input_size, dataset):
        self.input_size = input_size
        self.dataset = dataset

    def __getitem__(self, index):
        output = self.dataset[index]
        return {"data": output[0].reshape(self.input_size), "target": output[1]}
    
    def __len__(self,):
        return len(self.dataset)


class DictTensorDataset(TensorDataset):
    '''
    Normalize the sample of the dataset to be a dictionnary with keys "data" and "target" with correct size
    
    Attributes
    ----------
    input_size : tuple (channel, size_1, size_2, ...)
        Dimension of the input, channel is mandatory.
    
    tensors : torch.tensor/np.array
        The tensors that will be used to create the dataset
    '''
    def __init__(self, input_size, *tensors, **kwargs):
        super().__init__(*tensors)
        self.input_size = input_size

    def __getitem__(self, index):
        tensors = super().__getitem__(index)
        return {"data": tensors[0].reshape((-1, *self.input_size)), "target": tensors[1]}
    
    def __len__(self,):
        return super().__len__()
    

class CompleteDatasets():
    '''
    This class is a meta class to store any dataset that can be used for training, testing and validation.

    Attributes
    ----------
    dataset_train : torch.utils.data.Dataset
        Dataset used for training.
    dataset_test : torch.utils.data.Dataset
        Dataset used for testing.
    dataset_val : torch.utils.data.Dataset
        Dataset used for validation.
    dim_input : tuple (channel, size_1, size_2, ...) 
        Dimension of the input, channel is mandatory.
    dim_output : tuple (channel, size_1, size_2, ...)
        Dimension of the target if any, channel is mandatory.
    transform_back : function
        Function to transform the output of the model to the original space. If None, the identity is used.
        Can be useful for plotting images
    '''
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
    


