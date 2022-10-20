#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file is almost entirely copy-pasted from Boris Muzellec Repo : https://github.com/BorisMuzellec/MissingDataOT.git

import torch
import numpy as np

from scipy import optimize


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
        mask = RectangleMCAR_Mask(X, p = args['p_missing'], p_obs = args['p_obs'])
    elif args['missing_mechanism'] == 'none':
        mask = torch.zeros_like(X)[:,:1]
    else :
        raise ValueError("Missing mechanism not recognized")
    return mask, mask_Y


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



def RectangleMCAR_Mask(X, p, p_obs,type = 'rows'):
    """
    Missing completely at random but only with the top part of the image
    """
    n, c, dim = X.shape[0], X.shape[1], X.shape[2:]
    total_dim = np.prod(dim).astype(int)



    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)
    mask = torch.zeros(n, 1, *dim).bool() if to_torch else np.zeros((n, 1, *dim)).astype(bool)

    ber = torch.rand(n,)
    if type == 'columns':
        idx_rectangle = int(p_obs * dim[-1]) ## Number of columns (ie : second dimension) that will all be observed
        while(len(ber.shape)<len(mask.shape)):
            ber = ber.unsqueeze(-1)
        ber = ber.expand(n, 1, *dim[:-1], idx_rectangle)
        mask[..., :idx_rectangle] = ber < p
    elif type == 'rows':
        idx_rectangle = int(p_obs * dim[0]) ## Number of rows (ie : first dimension) that will all be observed
        while(len(ber.shape)<len(mask.shape)):
            ber = ber.unsqueeze(-1)
        ber = ber.expand(n, 1, idx_rectangle, *dim[1:],)
        mask[:,:, :idx_rectangle, :] = ber < p

    return mask