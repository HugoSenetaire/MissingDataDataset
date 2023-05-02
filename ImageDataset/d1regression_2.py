import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import math
import scipy.stats
from torch.utils.data.dataset import TensorDataset
import pickle

def predump(root_dir):
    lst = []
    np.random.seed(41)

    size = 1000

    points = np.random.beta(0.5,1,8*size//10)*5+0.5

    np.random.shuffle(points)
    lst += points.tolist()
    zones = [[len(lst),'Asymmetric']]

    points = 3*np.cos(np.linspace(0,5,num=size))-2
    points = points+np.random.normal(scale=np.abs(points)/4,size=size)
    lst += points.tolist()
    zones += [[len(lst),'Symmetric']]

    lst += [np.random.uniform(low=i,high=j)
            for i,j in zip(np.linspace(-2,-4.5,num=size//2),
                        np.linspace(-0.5,9.,num=size//2))]

    zones += [[len(lst),'Uniform']]

    points = np.r_[8+np.random.uniform(size=size//2)*0.5,
                1+np.random.uniform(size=size//2)*3.,
                -4.5+np.random.uniform(size=-(-size//2))*1.5]

    np.random.shuffle(points)

    lst += points.tolist()
    zones += [[len(lst),'Multimodal']]

    y_train_synthetic = np.array(lst).reshape(-1,1)
    x_train_synthetic = np.arange(y_train_synthetic.shape[0]).reshape(-1,1)
    x_train_synthetic = x_train_synthetic/x_train_synthetic.max()

    disord = np.arange(y_train_synthetic.shape[0])
    np.random.shuffle(disord)

    x_train_synthetic = x_train_synthetic[disord]
    y_train_synthetic = y_train_synthetic[disord]

    # Train = 45%, Validation = 5%, Test = 50%

    x_test_synthetic = x_train_synthetic[:x_train_synthetic.shape[0]//2].reshape(-1,1)
    y_test_synthetic = y_train_synthetic[:x_train_synthetic.shape[0]//2].reshape(-1,1)
    y_train_synthetic = y_train_synthetic[x_train_synthetic.shape[0]//2:].reshape(-1,1)
    x_train_synthetic = x_train_synthetic[x_train_synthetic.shape[0]//2:].reshape(-1,1)

    x_valid_synthetic = x_train_synthetic[:x_train_synthetic.shape[0]//10]
    y_valid_synthetic = y_train_synthetic[:x_train_synthetic.shape[0]//10]
    y_train_synthetic = y_train_synthetic[x_train_synthetic.shape[0]//10:]
    x_train_synthetic = x_train_synthetic[x_train_synthetic.shape[0]//10:]

    plt.figure(figsize=(15,7))

    plt.plot(x_valid_synthetic,y_valid_synthetic,'o',label='validation points')
    plt.plot(x_train_synthetic,y_train_synthetic,'o',label='training points',alpha=0.2)
    plt.plot(x_test_synthetic,y_test_synthetic,'o',label='testing points',alpha=0.2)
    for i in range(len(zones)):
        if i!= len(zones)-1:
            plt.axvline(x=zones[i][0]/len(lst),linestyle='--',c='grey')
        if i==0:
            plt.text(x=(zones[i][0])/(2*len(lst)),y=y_train_synthetic.min()-0.5,
                    s=zones[i][1], horizontalalignment='center', fontsize=20, color='grey')
        else:
            plt.text(x=(zones[i-1][0]+zones[i][0])/(2*len(lst)),y=y_train_synthetic.min()-0.5,
                    s=zones[i][1], horizontalalignment='center', fontsize=20, color='grey')

    plt.legend(loc="lower left", bbox_to_anchor=(0.,0.1))
    plt.savefig(os.path.join(root_dir,"data.png"))

    print(x_train_synthetic.shape)
    print(x_valid_synthetic.shape)
    print(x_test_synthetic.shape)

    with open(os.path.join(root_dir,"x_train.pkl"), "wb") as file:
        pickle.dump(x_train_synthetic, file)
    with open(os.path.join(root_dir,"y_train.pkl"), "wb") as file:
        pickle.dump(y_train_synthetic, file)

    with open(os.path.join(root_dir,"x_val.pkl"), "wb") as file:
        pickle.dump(x_valid_synthetic, file)
    with open(os.path.join(root_dir,"y_val.pkl"), "wb") as file:
        pickle.dump(y_valid_synthetic, file)

    with open(os.path.join(root_dir,"x_test.pkl"), "wb") as file:
        pickle.dump(x_test_synthetic, file)
    with open(os.path.join(root_dir,"y_test.pkl"), "wb") as file:
        pickle.dump(y_test_synthetic, file)


class d1Regression_2():
    def __init__(self,
            root_dir: str,
            transform = None,
            target_transform = None,
            download: bool = False,
            seed = None,
            **kwargs,):
        root_dir = os.path.join(root_dir, "1dregression_2")
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        x_train_path = os.path.join(root_dir,"x_train.pkl")
        y_train_path = os.path.join(root_dir,"y_train.pkl")
        x_val_path = os.path.join(root_dir,"x_val.pkl")
        y_val_path = os.path.join(root_dir,"y_val.pkl")
        x_test_path = os.path.join(root_dir,"x_test.pkl")
        y_test_path = os.path.join(root_dir,"y_test.pkl")

        for path in [x_train_path,y_train_path,x_val_path,y_val_path,x_test_path,y_test_path]:
            if not os.path.exists(path):
                print("Generating data for 1dregression_2 because {} doesn't exist".format(path))
                self.generate_data(root_dir)
        with open(x_train_path, "rb") as file:
            self.x_train = pickle.load(file)
        with open(y_train_path, "rb") as file:
            self.y_train = pickle.load(file)
        with open(x_val_path, "rb") as file:
            self.x_val = pickle.load(file)
        with open(y_val_path, "rb") as file:
            self.y_val = pickle.load(file)
        with open(x_test_path, "rb") as file:
            self.x_test = pickle.load(file)
        with open(y_test_path, "rb") as file:
            self.y_test = pickle.load(file)

        self.dataset_train = TensorDataset(torch.from_numpy(self.x_train).float(),torch.from_numpy(self.y_train).float())
        self.dataset_val = TensorDataset(torch.from_numpy(self.x_val).float(),torch.from_numpy(self.y_val).float())
        self.dataset_test = TensorDataset(torch.from_numpy(self.x_test).float(), torch.from_numpy(self.y_test).float())
      
    def get_dim_input(self,):
        return (1,1)

    def get_dim_output(self,):
        return (1,)