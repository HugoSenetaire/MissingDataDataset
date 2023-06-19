from .abstract_stats_mask_generator import AbstractStatsGenerator
from ..utils import pick_coeffs, fit_intercepts

import torch
import numpy as np


class MARMaskGenerator(AbstractStatsGenerator):

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
    def __init__(self, p, p_obs, accross_channel = True):
        super().__init__(accross_channel = accross_channel)
        self.p = p
        self.p_obs = p_obs
 

    def calculate_stats(self, dataset, nb_samples=10000):
        """
        Calculate the statistics of the model on the dataset.
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset on which to calculate the statistics.
            Should follow the usual structure here {'data': data, 'target': target}
        """
        indexes = np.random.choice(len(dataset), min(nb_samples, len(dataset)))
        data = torch.cat([dataset.__getitem__(i)['data'].unsqueeze(0) for i in indexes], 0)

        if self.accross_channel:
            data_per_channel = data.flatten(2).sum(1)
        else:
            data_per_channel = data.flatten(1)

        
        d = data_per_channel.shape[1]
        self.d_obs = max(int(self.p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
        self.d_na = d - self.d_obs ## number of variables that will have missing values

        ### Sample variables that will all be observed, and those with missing values:
        self.idxs_obs = np.random.choice(d, self.d_obs, replace=False)
        self.idxs_nas = np.array([i for i in range(d) if i not in self.idxs_obs])

        ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
        ### The parameters of this logistic model are random.

        ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
        self.coeffs = pick_coeffs(data_per_channel)
        ### Pick the intercepts to have a desired amount of missing values
        self.intercepts = fit_intercepts(data_per_channel, self.coeffs, self.p)
        self.stats_calculated = True


    def masking_rule(self, batch):
        data = batch['data']
        batch_size = data.shape[0]
        if self.accross_channel:
            mask_size = (1, *data.shape[2:])
            data_per_channel = data.flatten(2).sum(1)
        else:
            mask_size = data.shape[1:]
            data_per_channel = data.flatten(1)

        mask = torch.zeros(batch_size, mask_size).int().flatten(1)
        ps = torch.sigmoid(data_per_channel[:, self.idxs_obs].mm(self.coeffs) + self.intercepts)
        ber = torch.rand(batch_size, self.d_na)
        mask[:, self.idxs_nas] = ber > ps
        mask = mask.reshape(batch_size, *mask_size).int()
        return mask



