# train density estimators on various datasets



from .datasets import BSDS300, CIFAR10, GAS, HEPMASS, MINIBOONE, MNIST, POWER
from ...complete_dataset import DictTensorDataset
import torch
import numpy as np
dic_maf_dataset = {
    "bsds300_maf" : BSDS300,
    "cifar10_maf" : CIFAR10,
    "gas_maf" : GAS,
    "hepmass_maf" : HEPMASS,
    "miniboone_maf" : MINIBOONE,
    "mnist_maf" : MNIST,
    "power_maf" : POWER,
}




def get_maf_dataset(name, root):
    """
    Loads the dataset. Has to be called before anything else.
    :param name: string, the dataset's name
    """

    assert isinstance(name, str), 'Name must be a string'
    assert name in dic_maf_dataset.keys(), f'Unknown dataset {name}'

    if name == 'mnist':
        data = MNIST(logit=True, dequantize=True)
        data_name = name

    if name == 'bsds300_maf':
        data = BSDS300(root=root)

    elif name == 'cifar10_maf':
        data = CIFAR10(logit=True, flip=True, dequantize=True)
        data_name = name

    elif name == 'power_maf':
        data = POWER(root=root)

    elif name == 'gas_maf':
        data = GAS(root=root)

    elif name == 'hepmass_maf':
        data = HEPMASS(root=root)

    elif name == 'miniboone_maf':
        data = MINIBOONE(root=root)

    else:
        raise ValueError('Unknown dataset')
    
    dataset = MAFDataset(data)

    return dataset



class MAFDataset():
    def __init__(self,
            data,
            ):
        self.train_x = data.trn.x   
        self.val_x = data.val.x
        self.test_x = data.tst.x
        self.transform_back = None
        self.dim_input = self.train_x.shape[1:]
        if len(self.dim_input) == 1 or len(self.dim_input) == 2:
            self.dim_input = (1, *self.dim_input)
        self.dataset_train = DictTensorDataset(self.dim_input, torch.from_numpy(self.train_x), torch.zeros(self.train_x.shape[0]))
        self.dataset_test = DictTensorDataset(self.dim_input, torch.from_numpy(self.test_x), torch.zeros(self.test_x.shape[0]))
        self.dataset_val = DictTensorDataset(self.dim_input, torch.from_numpy(self.val_x), torch.zeros(self.val_x.shape[0]))
      
    def get_dim_input(self,):
        return self.dim_input

    def get_dim_output(self,):
        return (1,)
    
    