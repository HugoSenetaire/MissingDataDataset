from .abstract_stats_mask_generator import AbstractStatsGenerator
from ..utils import pick_coeffs, fit_intercepts

import torch
import numpy as np


class MNARMaskLogistics(AbstractStatsGenerator):

    """
    Missing at  mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.
    Attributes
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values.
    requires_stats : bool
        Need the full dataset to get the statistics from the model.
    
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """
    def __init__(self, p, p_params, exclude_inputs=True, accross_channel = True):
        super().__init__(accross_channel = accross_channel)
        self.p = p
        self.p_params = p_params
        self.exclude_inputs = True


    def calculate_stats(self, dataset, nb_samples=10000):
        """
        Calculate the statistics of the model on the dataset.
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset on which to calculate the statistics.
        """
        indexes = np.random.choice(len(dataset), min(nb_samples, len(dataset)))
        data = torch.cat([dataset.__getitem__(i)['data'].unsqueeze(0) for i in indexes], 0)

        if self.accross_channel:
            data_per_channel = data.flatten(2).sum(1)
        else:
            data_per_channel = data.flatten(1)

        
        d = data_per_channel.shape[1]

        self.d_params = max(int(self.p_params * d), 1) if self.exclude_inputs else d ## number of variables used as inputs (at least 1)
        self.d_na = d - self.d_params if self.exclude_inputs else d ## number of variables masked with the logistic model

        ### Sample variables that will be parameters for the logistic regression:
        self.idxs_params = np.random.choice(d, self.d_params, replace=False) if self.exclude_inputs else np.arange(d)
        self.idxs_nas = np.array([i for i in range(d) if i not in self.idxs_params]) if self.exclude_inputs else np.arange(d)

        ### Other variables will have NA proportions selected by a logistic model
        ### The parameters of this logistic model are random.

        ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
        self.coeffs = pick_coeffs(data_per_channel, self.idxs_params, self.idxs_nas)
        ### Pick the intercepts to have a desired amount of missing values
        self.intercepts = fit_intercepts(data_per_channel[:, self.idxs_params], self.coeffs, self.p)
        self.stats_calculated = True


    def masking_rule(self, batch):

        data = batch['data']
        batch_size = data.shape[0]

        if self.accross_channel:
            mask_size = (data.shape[0], 1, *data.shape[2:])
            data_per_channel = data.flatten(2).sum(1)
        else:
            mask_size = data.shape
            data_per_channel = data.flatten(1)

        mask = torch.ones(mask_size).int().flatten(1)

        ps = torch.sigmoid(data_per_channel[:, self.idxs_params].mm(self.coeffs) + self.intercepts)

        ber = torch.rand(batch_size, self.d_na)
        mask[:, self.idxs_nas] = ber > ps

        ## If the inputs of the logistic model are excluded from MNAR missingness,
        ## mask some values used in the logistic model at random.
        ## This makes the missingness of other variables potentially dependent on masked values
        if self.exclude_inputs:
            mask[:, self.idxs_params] = torch.rand(batch_size, self.d_params) > self.p
        mask = mask.reshape(mask_size).expand(data.shape)

        return mask
 

class MNARMaskSelfLogistics(AbstractStatsGenerator):
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
    def __init__(self,p, accross_channel=True):
        super().__init__(accross_channel)
        self.p = p


    def calculate_stats(self, dataset, nb_samples=10000):
        """
        Calculate the statistics of the model on the dataset.
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset on which to calculate the statistics.
        """
        indexes = np.random.choice(len(dataset), min(nb_samples, len(dataset)))
        data = torch.cat([dataset.__getitem__(i)['data'].unsqueeze(0) for i in indexes], 0)


        if self.accross_channel:
            data_per_channel = data.flatten(2).sum(1)
        else:
            data_per_channel = data.flatten(1)

        ### Variables will have NA proportions that depend on those observed variables, through a logistic model
        ### The parameters of this logistic model are random.

        ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
        self.coeffs = pick_coeffs(data_per_channel, self_mask=True)
        ### Pick the intercepts to have a desired amount of missing values
        self.intercepts = fit_intercepts(data_per_channel, self.coeffs, self.p, self_mask=True)
        self.stats_calculated = True


    def masking_rule(self, batch):

        data = batch['data']
        batch_size = data.shape[0]

        if self.accross_channel:
            mask_size = (data.shape[0], 1, *data.shape[2:])
            data_per_channel = data.flatten(2).sum(1)
        else:
            mask_size = data.shape
            data_per_channel = data.flatten(1)

        mask = torch.ones(mask_size).int().flatten(1)
        d = data_per_channel.shape[1]
        ps = torch.sigmoid(data_per_channel * self.coeffs + self.intercepts)
        ber = torch.rand(batch_size, d)
        mask = (ber > ps).int()
        mask = mask.reshape(mask_size).expand(data.shape)

        return mask