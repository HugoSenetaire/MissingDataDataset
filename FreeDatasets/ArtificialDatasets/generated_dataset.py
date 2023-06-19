#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is starting from Boris Muzellec Repo : https://github.com/BorisMuzellec/MissingDataOT.git
from sklearn.datasets import make_moons, make_swiss_roll, make_s_curve
from .toy_plane_dataset import SignDataset, SineWaveDataset, CheckerboardDataset, CrescentCubedDataset, CrescentDataset, FourCirclesDataset
import os
import pandas as pd
import wget
import numpy as np


GENERATED_DATASETS = ['multivariate_gaussian', 'linear_manually', 'swiss_roll', 's_curve', 'sign', 'sine_wave', 'checkerboard', \
            'crescent_cubed', 'crescent', 'four_circles', 'funnel_2d', 'pinwheel_dataset',]



def generated_dataset_loader(dataset, args):
    """
    Data loading utility for generated datasets. 

    Parameters
    ----------

    dataset : str
        Name of the dataset to retrieve.
        Valid values: see DATASETS.
        
    Returns
    ------
    X : ndarray
        Data values (predictive values only).
    Y : ndarray
        Target values.
    parameters : dict
        Parameters of the dataset (e.g. mean and covariance for multivariate gaussian) that can be useful for evaluation.
    """
    assert dataset in GENERATED_DATASETS , f"Dataset not supported: {dataset}"

    parameters = None

    if dataset in GENERATED_DATASETS:
        if dataset == 'moon':
            data = fetch_moon()
        elif dataset == 's_curve':
            data = fetch_s_curve()
        elif dataset == 'swiss_roll':
            data = fetch_swiss_roll()
        elif dataset == 'sin_wave':
            data = fetch_sine_wave()
        # elif dataset == 'sign':
        #     data = fetch_sign()
        elif dataset == 'checkerboard':
            data = fetch_checkerboard()
        elif dataset == 'crescent_cubed':
            data = fetch_crescent_cubed_dataset()
        elif dataset == 'crescent':
            data = fetch_crescent_dataset()
        elif dataset == 'four_circles':
            data = fetch_fourcircles()
        elif dataset == 'multivariate_gaussian':
            data, parameters = fetch_multivariate_gaussian(dim = args['dim'], rho=args['rho'])
        elif dataset == 'linear_manually':
            data, parameters = fetch_linear_manually(dim = args['dim'], noise = args['noise'], problem_type = args['problem_type'], min_weight=args['min_weight'], max_weight=args['max_weight'])
        elif dataset == 'funnel_2d':
            data, parameters = fetch_funnel(dim = 2)
        elif dataset == 'pinwheel_dataset':
            data, parameters = make_pinwheel_data(0.3, 0.05, 5, 2000, 0.25)

        X = data['data']
        print("MEAN HERE", np.mean(X, axis=0))
        Y = data['target']
        return X, Y, parameters

def fetch_linear_manually(dim = 10, noise = 0.1, problem_type = 'regression', min_weight = 1e-2, max_weight = 1e2, size = 10000):
    # weights = np.arange(min_weight, max_weight, (max_weight - min_weight)/dim) * (np.random.randint(2, size=dim)*2 - 1)
    weights = np.logspace(np.log10(min_weight), np.log10(max_weight), dim) * (np.random.randint(2, size=dim)*2 - 1) # Absolute value times sign
    X = np.random.rand(10000, dim)
    if problem_type == 'regression':
        Y = X @ weights + np.random.normal(0, noise, size = 10000)
    elif problem_type == 'classification':
        Y = (np.sign(X @ weights + np.random.normal(0, noise, size = 10000)) + 1) / 2
    else:
        raise ValueError(f"Problem type not supported: {problem_type}")
    parameters = {'weights': weights, 'bias': np.zeros_like(X[0])}
    return {'data': X, 'target': Y}, parameters 

def fetch_funnel(dim = 2):
    y = np.random.normal(0, 1, size = (10000,1 ))
    x = np.random.normal(np.zeros_like(y), np.exp(y), size = (10000, dim - 1))
    X = np.concatenate([x, y], axis = 1)
    return {'data': X, 'target': np.zeros_like(X[:, 0])}, None

def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    # code from Johnson et. al. (2016)
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    np.random.seed(1)

    features = np.random.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    feats = 10 * np.einsum('ti,tij->tj', features, rotations)

    data = np.random.permutation(np.hstack([feats, labels[:, None]]))
    current_data = {'data': data[:, 0:2], 'target' : labels}
    return current_data, None




def fetch_multivariate_gaussian(dim =10, rho = 0.5): # TODO @hhjs : Check with Toeplitz for instance to get easier covariance, par block ...
    # diag = np.exp(np.random.normal(0, 1.0, (dim, )))
    # lower_triangle = np.random.normal(loc = 0, scale = 1.0, size = (int(dim*(dim-1)//2), ))
    # aux = np.diag(diag)
    # aux[np.tril_indices(dim, -1)] = lower_triangle
    # cov =np.matmul(aux, aux.T)
    cov = np.zeros((dim, dim))
    for k in range(0, dim):
        for i in range(0, dim):
            cov[i, k] = np.power(rho, abs(i - k))
    print("Covariance matrix : \n", cov)
    print("VP", np.linalg.eigvals(cov))
    mean = np.random.normal(0, 1.0, (dim,))
    data = np.random.multivariate_normal(mean=mean, cov=cov, size=10000)
    print("MEAN", mean)
    print("MEAN EVALUATE", data.mean(axis=0))
    target = np.zeros(data.shape[0])
    data = {'data': data, 'target': target}
    parameters = {'mean': mean, 'cov': cov}
    return data, parameters




def fetch_moon():
    """
    Create a moon dataset similar to the one in "HOW TO DEAL WITH MISSING DATA IN SUPERVISED DEEP LEARNING?"
    """
    data, target = make_moons(n_samples=10000, noise=0.1, random_state=0)
    return {'data': data, 'target': target}
    
def fetch_swiss_roll():
    """
    Create a swiss roll dataset
    """
    data, target = make_swiss_roll(n_samples=10000, noise=0.5, random_state=0)
    data = data[:, [0, 2]]
    return {'data': data, 'target': target}


def fetch_s_curve():
    """
    Create a s curve dataset
    """
    data, target = make_s_curve(n_samples=10000, noise=0.3, random_state=0)
    data = data[:, [0, 2]]
    return {'data': data, 'target': target}

def fetch_sine_wave():
    """
    Create a sine wave dataset
    """
    dataset = SineWaveDataset(num_points=10000,)
    dataset.reset()
    data = dataset.data.numpy()
    target = np.zeros(len(data))
    return {'data': data, 'target': target}

def fetch_crescent_dataset():
    """
    Create a crescent dataset
    """
    dataset = CrescentDataset(num_points=10000,)
    dataset.reset()
    data = dataset.data.numpy()
    target = np.zeros(len(data))
    return {'data': data, 'target': target}

def fetch_crescent_cubed_dataset():
    """
    Create a crescent cube dataset
    """
    dataset = CrescentCubedDataset(num_points=10000,)
    dataset.reset()
    data = dataset.data.numpy()
    target = np.zeros(len(data))
    return {'data': data, 'target': target}

def fetch_checkerboard():
    """
    Create a checkerboard dataset
    """
    dataset = CheckerboardDataset(num_points=10000,)
    dataset.reset()
    data = dataset.data.numpy()
    target = np.zeros(len(data))
    return {'data': data, 'target': target}

def fetch_fourcircles():
    """
    Create a four circles dataset
    """
    dataset = FourCirclesDataset(num_points=10000,)
    dataset.reset()
    data = dataset.data.numpy()
    target = np.zeros(len(data))
    return {'data': data, 'target': target}
