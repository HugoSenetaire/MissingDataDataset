import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import math
import scipy.stats

import pickle

def predump(root_dir):
    x = np.random.uniform(low=-3.0, high=3.0, size=(4000, ))
    x = x.astype(np.float32)
    
    y = []
    for x_value in x:
        if x_value < 0:
            component = np.random.randint(low=1, high=6) # (1, 2, 3, 4, 5 with 0.5 prob)
    
            if component in [1, 2, 3, 4]:
                mu_value = np.sin(x_value)
                sigma_value = 0.15*(1.0/(1 + 1))
            elif component == 5:
                mu_value = -np.sin(x_value)
                sigma_value = 0.15*(1.0/(1 + 1))
    
            y_value = np.random.normal(mu_value, sigma_value)
        else:
            y_value = np.random.lognormal(0.0, 0.25) - 1.0
    
        y.append(y_value)
    y = np.array(y, dtype=np.float32)
    
    with open(os.path.join(root_dir,"x.pkl"), "wb") as file:
        pickle.dump(x, file)
    with open(os.path.join(root_dir,"y.pkl"), "wb") as file:
        pickle.dump(y, file)
    
    
    
    num_samples = 2048
    x = np.linspace(-3.0, 3.0, num_samples, dtype=np.float32)
    y_samples = np.linspace(-3.0, 3.0, num_samples) # (shape: (num_samples, ))
    x_values_2_scores = {}
    for x_value in x:
        if x_value < 0:
            scores = 0.8*scipy.stats.norm.pdf(y_samples, np.sin(x_value), 0.15*(1.0/(1 + 1))) + 0.2*scipy.stats.norm.pdf(y_samples, -np.sin(x_value), 0.15*(1.0/(1 + 1)))
        else:
            scores = scipy.stats.lognorm.pdf(y_samples+1.0, 0.25)
    
        x_values_2_scores[x_value] = scores
    
    with open(os.path.join(root_dir,"gt_x_values_2_scores.pkl"), "wb") as file:
        pickle.dump(x_values_2_scores, file)


class d1Regression_1():
    def __init__(self,
            root_dir: str,
            transform = None,
            target_transform = None,
            download: bool = False,
            seed = None,
            **kwargs,):
    
        root_dir = os.path.join(root_dir, "1dregression_1")
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        path_x = os.path.join(root_dir,"x.pkl")
        path_y = os.path.join(root_dir,"y.pkl")
        if not os.path.exists(path_x) or not os.path.exists(path_y):
            print(f"{path_x} or {path_y} not found")
            predump(root_dir=root_dir)
        with open(path_x, "rb") as file:
            self.x = pickle.load(file)
        with open(path_y, "rb") as file:
            self.y = pickle.load(file)
       
        self.train_x, self.test_x, self.val_x = self.x[:2000].reshape(-1,1), self.x[2000:3000].reshape(-1,1), self.x[3000:].reshape(-1,1)
        self.train_y, self.test_y, self.val_y = self.y[:2000].reshape(-1,1), self.y[2000:3000].reshape(-1,1), self.y[3000:].reshape(-1,1)
        
        self.dataset_train = TensorDataset(torch.from_numpy(self.train_x), torch.from_numpy(self.train_y))
        self.dataset_test = TensorDataset(torch.from_numpy(self.test_x), torch.from_numpy(self.test_y))
        self.dataset_val = TensorDataset(torch.from_numpy(self.val_x), torch.from_numpy(self.val_y))
      
    def get_dim_input(self,):
        return (1,1)

    def get_dim_output(self,):
        return (1,)