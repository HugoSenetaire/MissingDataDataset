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
    current_X = X[:,0,].flatten(1)
    if seed is not None :
        np.random.seed(seed = seed)
    if args['missing_mechanism'] == 'mcar':
        mask = MCAR_mask(current_X, p = args['p_missing'], p_obs = args['p_obs'])
    elif args['missing_mechanism'] == 'none':
        mask = torch.zeros_like(X)
    elif args['missing_mechanism'] == 'mar':
        mask = MAR_mask(current_X, p = args['p_missing'], p_obs = args['p_obs'])
    elif args['missing_mechanism'] == 'mnar_logistic':
        mask = MNAR_mask_logistic(current_X, p = args['p_missing'], p_params = args['p_params'], exclude_inputs=args['exclude_inputs'])
    elif args['missing_mechanism'] == 'mnar_self_logistic':
        mask = MNAR_self_mask_logistic(current_X, p = args['p_missing'])
    elif args['missing_mechanism'] == 'mnar_quantiles':
        mask = MNAR_mask_quantiles(current_X, p = args['p_missing'], q = args["quantile"], p_obs =  args['p_obs'], cut=args['cut'], MCAR=False)
    else:
        raise ValueError('Unknown missing mechanism for tabular data: {}'.format(args['missing_mechanism']))
    mask = mask.reshape((X.shape[0], 1, *X.shape[2:]))
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

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = int(p_obs * d) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < p

    return mask





##### Missing At Random ######

def MAR_mask(X, p, p_obs):
    """
    Missing at  mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask

##### Missing not at random ######

def MNAR_mask_logistic(X, p, p_params =.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
    d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask

def MNAR_self_mask_logistic(X, p):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
    mask = ber < ps if to_torch else ber < ps.numpy()

    return mask


def MNAR_mask_quantiles(X, p, q, p_obs, cut='both', MCAR=False):
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    q : float
        Quantile level at which the cuts should occur
    p_obs : float
        Proportion of variables that will have missing values
    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.
        
    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.
        
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """
    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_na = max(int(p_obs * d), 1) ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(d, d_na, replace=False) ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == 'upper':
        quants = quantile(X[:, idxs_na], 1-q, dim=0)
        m = X[:, idxs_na] >= quants
    elif cut == 'lower':
        quants = quantile(X[:, idxs_na], q, dim=0)
        m = X[:, idxs_na] <= quants
    elif cut == 'both':
        u_quants = quantile(X[:, idxs_na], 1-q, dim=0)
        l_quants = quantile(X[:, idxs_na], q, dim=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
    ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d, dtype = torch.float32)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na, dtype = torch.float32)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts