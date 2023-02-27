#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file is almost entirely copy-pasted from Boris Muzellec Repo : https://github.com/BorisMuzellec/MissingDataOT.git

import torch
import numpy as np

from scipy import optimize
from .mask_generators import ImageMaskGenerator


#### MASK LOADER :

#TODO (@hhjs): Maybe it would be more interesting to work in a similar way to pytorch transform in torchvision to combine the masks
def mask_loader_image(X, Y, args, seed = None):
    """
    Loads the mask for the missing values according to the missing mechanism specified in args.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    Y : torch.DoubleTensor or np.ndarray, shape (n, d)
        Target for which missing values will be simulated.
    args : dict
        Dictionary of arguments.
        args['missing_mechanism'] : str
            Name of the missing mechanism to use.
            Possible values are:
            ["mcar", "dual_mask", "dual_mask_opposite", "mar", "mnar_logistic", "mnar_self_logistic", "mnar_quantiles"]
                - 'mcar': missing completely at random.
                - 'mar': missing at random
                - 'dual_mask' : MCAR with only two schemes of masks
                - 'mnar_logistic': missing not at random, with a logistic masking model.
                - 'mnar_self_logistic': missing not at random, with a logistic masking model,
                    where the coefficients are estimated from the data.
                - 'mnar_quantiles': missing not at random, with a quantile masking model.
                - 'none' : no missing data
        args['p_missing'] : float
            Proportion of missing values to generate for variables which will have missing values.
        args['p_obs'] : float
            Proportion of variables with *no* missing values.
        args['p_param] : float
            Proportion of variables with *no* missing values that will be used for the logistic masking model. Only if exclue input.
        args['exclude_inputs'] : bool
            If True, the input variables will not be used for the logistic masking model for the MNAR logistic mask.
        args['quantiles'] : float
            Quantile level at which the cuts should occur (for mnar quantile)
        args['cut]: str 
            Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.
        Options are :
            - 'lower' : cut at the lower quantile
            - 'upper' : cut at the upper quantile
            - "both" : cut at both quantiles

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing). #TODO : That's the opposite of what I've been doing for a year, do you mind if I change it ? @hhjs
    mask_Y : torch.BoolTensor or np.ndarray (depending on type of Y)
        Mask of generated missing values (True if the value is missing).
    """
    mask_Y = None
    if seed is not None :
        np.random.seed(seed = seed)
    if args['missing_mechanism'] == 'mcar':
        mask = MCAR_mask(X, p = args['p_missing'], p_obs = args['p_obs'])
    elif args['missing_mechanism'] == 'rectangle_mcar':
        mask = RectangleMCAR_Mask(X, p = args['p_missing'], p_obs = args['p_obs'], orientation = args['orientation'])
    elif args['missing_mechanism'] == 'mar_mnist':
        mask = MAR_MNIST_mask(X,)
    elif args['missing_mechanism'] == 'none':
        mask = torch.zeros_like(X)[:,:1]
    elif args['missing_mechanism'] == 'mask_generators_vaeac':
        mask = mask_generators_vaeac(X, Y, args)
    elif args['missing_mechanism'] == 'target_correlated':
        mask = mask_target_correlated(X, Y, args)
    else :
        raise ValueError("Missing mechanism not recognized {}".format(args['missing_mechanism']))

    if mask is not None :
        mask = mask.expand(X.shape)
    return mask, mask_Y, None


########

def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.
    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.
    q : float
        Quantile level (starting from lower values).
    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.
    Returns
    -------
        quantiles : torch.DoubleTensor
    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]





##################### MISSING DATA MECHANISMS #############################

def MCAR_mask(X, p, p_obs):
    """
    Missing completely at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according completely randomly.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, c, dim = X.shape[0], X.shape[1], X.shape[2:]
    total_dim = np.prod(dim).astype(int)


    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, total_dim).bool() if to_torch else np.zeros((n, total_dim)).astype(bool)

    d_obs = int(p_obs * total_dim) ## number of variables that will have no missing values (at least one variable)
    d_na = total_dim - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(total_dim, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(total_dim) if i not in idxs_obs])

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < p

    mask = mask.reshape(n, 1, *dim)

    return mask

def mask_target_correlated(X, Y, args):
    """
    Missing at random mask where the missingness rate is defined by the target.
    """
    nb_targets = len(np.unique(Y))
    rates = np.linspace(0.1, 0.9, nb_targets)

    n, c, dim = X.shape[0], X.shape[1], X.shape[2:]
    ber = torch.rand(n, 1, *dim)
    p = torch.tensor([rates[int(y)] for y in Y], device = X.device).reshape(-1, 1, 1, 1).expand(n, 1, *dim)

    mask = ber<p
    mask = mask.reshape(n, 1, *dim)

    return mask

def mask_generators_vaeac(X, Y, args):
    """
    Missing completely at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according completely randomly.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        
    """
    mask_generator = ImageMaskGenerator()
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = mask_generator(X, )
    
    return mask


def RectangleMCAR_Mask(X, p, p_obs, orientation = 'rows'):
    """
    Missing completely at random but only with the top part of the image
    """

    if orientation == 'columns':
        X = X.permute(0,1,3,2)
    n, c, dim = X.shape[0], X.shape[1], X.shape[2:]
    total_dim = np.prod(dim).astype(int)



    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)
    mask = torch.zeros(n, 1, *dim).bool() if to_torch else np.zeros((n, 1, *dim)).astype(bool)

    ber = torch.rand(n,)
    idx_rectangle = int((1-p_obs)*dim[0])
    while(len(ber.shape)<len(mask.shape)):
        ber = ber.unsqueeze(-1)
    ber = ber.expand(n, 1, idx_rectangle, *dim[1:],)
    mask[:,:, :idx_rectangle, :] = ber < p


    if orientation == 'columns':
        mask = mask.permute(0,1,3,2)
        X = X.permute(0,1,3,2)

    return mask

def MAR_MNIST_mask(X, type = 'rows'):
    """
    Used for MNIST in MIWAE (https://proceedings.mlr.press/v97/mattei19a.html),
    We consider a MAR version of MNIST where all bottom
    halves of the pixels are observed. For each digit, either
    the top half, top quarter, or second quarter, is missing
    (depending on the number of white pixels in the bottom half).
    """

    if type == 'columns':
        X = X.permute(0,1,3,2)

    n, c, dim = X.shape[0], X.shape[1], X.shape[2:]
    total_dim = np.prod(dim).astype(int)

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)
    mask = torch.zeros(n, 1, *dim).bool() 


    ber1 = torch.rand(n,)
    ber2 = torch.rand(n,)
    pi_x = torch.sigmoid(X[:,:, dim[0]//2 :,].flatten(1).sum(-1)) / total_dim + 0.4
    idx_half = int(dim[0]//2)
    idx_quarter = int(dim[0]//4)
    h = (ber1 < pi_x).to(torch.int64) + (ber2 < pi_x).to(torch.int64)

    while(len(h.shape)<len(mask.shape)):
        h = h.unsqueeze(-1)
    h = h.expand(mask.shape)
    mask[:,:, idx_half:, :] = False
    mask[:,:, :idx_quarter,] = (h[:,:, :idx_quarter,] != 0)
    mask[:,:, idx_quarter:idx_half,] = (h[:,:, idx_quarter:idx_half,] != 2)

    if type == 'columns':
        mask = mask.permute(0,1,3,2)
        X = X.permute(0,1,3,2)

    return mask
