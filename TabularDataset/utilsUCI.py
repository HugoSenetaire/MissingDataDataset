#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is starting from Boris Muzellec Repo : https://github.com/BorisMuzellec/MissingDataOT.git
from sklearn.datasets import load_iris, load_wine, load_boston, fetch_california_housing, make_moons
import os
import pandas as pd
import wget
import numpy as np


DATASETS_TENSOR = ['iris', 'wine', 'boston', 'california', 'parkinsons', \
            'climate_model_crashes', 'concrete_compression', \
            'yacht_hydrodynamics', 'airfoil_self_noise', \
            'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', \
            'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', \
            'blood_transfusion', 'breast_cancer_diagnostic', \
            'connectionist_bench_vowel', 'concrete_slump', \
            'wine_quality_red', 'wine_quality_white', 'moon', 'multivariate_gaussian', \
             'linear_manually']



def tensor_dataset_loader(dataset, args):
    """
    Data loading utility for a subset of UCI ML repository datasets. Assumes 
    datasets are located in './datasets'. If the called for dataset is not in 
    this folder, it is downloaded from the UCI ML repo.

    Parameters
    ----------

    dataset : str
        Name of the dataset to retrieve.
        Valid values: see DATASETS.
        
    Returns
    ------
    X : ndarray
        Data values (predictive values only).
    """
    assert dataset in DATASETS_TENSOR , f"Dataset not supported: {dataset}"

    if dataset in DATASETS_TENSOR:
        if dataset == 'iris':
            data = load_iris()
        elif dataset == 'wine':
            data = load_wine()
        elif dataset == 'boston':
            data = load_boston()
        elif dataset == 'california':
            data = fetch_california_housing()
        elif dataset == 'parkinsons':
            data = fetch_parkinsons()
        elif dataset == 'climate_model_crashes':
            data = fetch_climate_model_crashes()
        elif dataset == 'concrete_compression':
            data = fetch_concrete_compression()
        elif dataset == 'yacht_hydrodynamics':
            data = fetch_yacht_hydrodynamics()
        elif dataset == 'airfoil_self_noise':
            data = fetch_airfoil_self_noise()
        elif dataset == 'connectionist_bench_sonar':
            data = fetch_connectionist_bench_sonar()
        elif dataset == 'ionosphere':
            data = fetch_ionosphere()
        elif dataset == 'qsar_biodegradation':
            data = fetch_qsar_biodegradation()
        elif dataset == 'seeds':
            data = fetch_seeds()
        elif dataset == 'glass':
            data = fetch_glass()
        elif dataset == 'ecoli':
            data = fetch_ecoli()
        elif dataset == 'yeast':
            data = fetch_yeast()
        elif dataset == 'libras':
            data = fetch_libras()
        elif dataset == 'planning_relax':
            data = fetch_planning_relax()
        elif dataset == 'blood_transfusion':
            data = fetch_blood_transfusion()
        elif dataset == 'breast_cancer_diagnostic':
            data = fetch_breast_cancer_diagnostic()
        elif dataset == 'connectionist_bench_vowel':
            data = fetch_connectionist_bench_vowel()
        elif dataset == 'concrete_slump':
            data = fetch_concrete_slump()
        elif dataset == 'wine_quality_red':
            data = fetch_wine_quality_red()
        elif dataset == 'wine_quality_white':
            data = fetch_wine_quality_white()
        elif dataset == 'moon':
            data = fetch_moon()
        elif dataset == 'multivariate_gaussian':
            data = fetch_multivariate_gaussian(dim = args['dim'])
        elif dataset == 'linear_manually':
            data = fetch_linear_manually(dim = args['dim'], noise = args['noise_dataset'], problem_type = args['problem_type'], min_weight=args['min_weight'], max_weight=args['max_weight'])

        X = data['data']
        Y = data['target']
        return X, Y

def fetch_linear_manually(dim = 10, noise = 0.1, problem_type = 'regression', min_weight = 1e-2, max_weight = 1e2, size = 10000):
    weights = np.arange(min_weight, max_weight, (max_weight - min_weight)/dim) * (np.random.randint(2, size=dim)*2 - 1)
    X = np.random.rand(10000, dim)
    if problem_type == 'regression':
        Y = X @ weights + np.random.normal(0, noise, size = 10000)
    elif problem_type == 'classification':
        Y = (np.sign(X @ weights + np.random.normal(0, noise, size = 10000)) + 1) / 2
    else:
        raise ValueError(f"Problem type not supported: {problem_type}")
    print(weights)
    print(X.mean(), Y.mean())
    return {'data': X, 'target': Y}



def fetch_multivariate_gaussian(dim =10,): # TODO @hhjs : Check with Toeplitz for instance to get easier covariance, par block ...
    # diag = np.exp(np.random.normal(0, 1.0, (dim, )))
    # lower_triangle = np.random.normal(loc = 0, scale = 1.0, size = (int(dim*(dim-1)//2), ))
    # aux = np.diag(diag)
    # aux[np.tril_indices(dim, -1)] = lower_triangle
    # cov =np.matmul(aux, aux.T)
    rho = 0.5
    cov = np.zeros((dim, dim))
    for k in range(0, dim):
        for i in range(0, dim):
            cov[i, k] = np.power(rho, abs(i - k))
    print("Covariance matrix : \n", cov)
    print("VP", np.linalg.eigvals(cov))
    mean = np.random.normal(0, 1.0, (dim,))
    data = np.random.multivariate_normal(mean=mean, cov=cov, size=10000)
    target = np.zeros(data.shape[0])
    return {'data': data, 'target': target}




def fetch_moon():
    """
    Create a moon dataset similar to the one in "HOW TO DEAL WITH MISSING DATA IN SUPERVISED DEEP LEARNING?"
    """
    data, target = make_moons(n_samples=4000, noise=0.1, random_state=0)
    return {'data': data, 'target': target}
    



def fetch_parkinsons():
    if not os.path.isdir('local/datasets/parkinsons'):
        os.makedirs('datasets/parkinsons')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
        wget.download(url, out='datasets/parkinsons/')

    with open('datasets/parkinsons/parkinsons.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = 0)
        Xy = {}
        Xy['data'] = df.values[:, 1:].astype('float')
        Xy['target'] =  df.values[:, 0]

    return Xy


def fetch_climate_model_crashes():
    if not os.path.isdir('local/datasets/climate_model_crashes'):
        os.makedirs('datasets/climate_model_crashes')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat'
        wget.download(url, out='datasets/climate_model_crashes/')

    with open('datasets/climate_model_crashes/pop_failures.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = 0)
        Xy = {}
        Xy['data'] = df.values[:, 2:-1]
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_concrete_compression():
    if not os.path.isdir('local/datasets/concrete_compression'):
        os.makedirs('datasets/concrete_compression')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        wget.download(url, out='datasets/concrete_compression/')

    with open('datasets/concrete_compression/Concrete_Data.xls', 'rb') as f:
        df = pd.read_excel(io=f)
        Xy = {}
        Xy['data'] = df.values[:, :-2]
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_yacht_hydrodynamics():
    if not os.path.isdir('local/datasets/yacht_hydrodynamics'):
        os.makedirs('datasets/yacht_hydrodynamics')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
        wget.download(url, out='datasets/yacht_hydrodynamics/')

    with open('datasets/yacht_hydrodynamics/yacht_hydrodynamics.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] =  df.values[:, -1]

    return Xy

def fetch_airfoil_self_noise():
    if not os.path.isdir('local/datasets/airfoil_self_noise'):
        os.makedirs('datasets/airfoil_self_noise')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
        wget.download(url, out='datasets/airfoil_self_noise/')

    with open('datasets/airfoil_self_noise/airfoil_self_noise.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_connectionist_bench_sonar():
    if not os.path.isdir('local/datasets/connectionist_bench_sonar'):
        os.makedirs('datasets/connectionist_bench_sonar')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
        wget.download(url, out='datasets/connectionist_bench_sonar/')

    with open('datasets/connectionist_bench_sonar/sonar.all-data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_ionosphere():
    if not os.path.isdir('local/datasets/ionosphere'):
        os.makedirs('datasets/ionosphere')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        wget.download(url, out='datasets/ionosphere/')

    with open('datasets/ionosphere/ionosphere.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_qsar_biodegradation():
    if not os.path.isdir('local/datasets/qsar_biodegradation'):
        os.makedirs('datasets/qsar_biodegradation')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
        wget.download(url, out='datasets/qsar_biodegradation/')

    with open('datasets/qsar_biodegradation/biodeg.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_seeds():
    if not os.path.isdir('local/datasets/seeds'):
        os.makedirs('datasets/seeds')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
        wget.download(url, out='datasets/seeds/')

    with open('datasets/seeds/seeds_dataset.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_glass():
    if not os.path.isdir('local/datasets/glass'):
        os.makedirs('datasets/glass')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
        wget.download(url, out='datasets/glass/')

    with open('datasets/glass/glass.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_ecoli():
    if not os.path.isdir('local/datasets/ecoli'):
        os.makedirs('datasets/ecoli')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
        wget.download(url, out='datasets/ecoli/')

    with open('datasets/ecoli/ecoli.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy

def fetch_yeast():
    if not os.path.isdir('local/datasets/yeast'):
        os.makedirs('datasets/yeast')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
        wget.download(url, out='datasets/yeast/')

    with open('datasets/yeast/yeast.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_libras():
    if not os.path.isdir('local/datasets/libras'):
        os.makedirs('datasets/libras')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data'
        wget.download(url, out='datasets/libras/')

    with open('datasets/libras/movement_libras.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy

def fetch_planning_relax():
    if not os.path.isdir('local/datasets/planning_relax'):
        os.makedirs('datasets/planning_relax')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt'
        wget.download(url, out='datasets/planning_relax/')

    with open('datasets/planning_relax/plrx.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_blood_transfusion():
    if not os.path.isdir('local/datasets/blood_transfusion'):
        os.makedirs('datasets/blood_transfusion')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
        wget.download(url, out='datasets/blood_transfusion/')

    with open('datasets/blood_transfusion/transfusion.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy

def fetch_breast_cancer_diagnostic():
    if not os.path.isdir('local/datasets/breast_cancer_diagnostic'):
        os.makedirs('datasets/breast_cancer_diagnostic')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
        wget.download(url, out='datasets/breast_cancer_diagnostic/')

    with open('datasets/breast_cancer_diagnostic/wdbc.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 2:].astype('float')
        Xy['target'] =  df.values[:, 1]

    return Xy


def fetch_connectionist_bench_vowel():
    if not os.path.isdir('local/datasets/connectionist_bench_vowel'):
        os.makedirs('datasets/connectionist_bench_vowel')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data'
        wget.download(url, out='datasets/connectionist_bench_vowel/')

    with open('datasets/connectionist_bench_vowel/vowel-context.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 3:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_concrete_slump():
    if not os.path.isdir('local/datasets/concrete_slump'):
        os.makedirs('datasets/concrete_slump')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
        wget.download(url, out='datasets/concrete_slump/')

    with open('datasets/concrete_slump/slump_test.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, 1:-3].astype('float')
        Xy['target'] =  df.values[:, -3:]

    return Xy


def fetch_wine_quality_red():
    if not os.path.isdir('local/datasets/wine_quality_red'):
        os.makedirs('datasets/wine_quality_red')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
        wget.download(url, out='datasets/wine_quality_red/')

    with open('datasets/wine_quality_red/winequality-red.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_wine_quality_white():
    if not os.path.isdir('local/datasets/wine_quality_white'):
        os.makedirs('datasets/wine_quality_white')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
        wget.download(url, out='datasets/wine_quality_white/')

    with open('datasets/wine_quality_white/winequality-white.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy