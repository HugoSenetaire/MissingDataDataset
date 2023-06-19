from ..abstract_mask_generator import AbstractGenerator
from ..utils import quantile

import torch
import numpy as np


class MNARQuantile(AbstractGenerator):
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
    def __init__(self, p, q, p_obs, cut ='upper', accross_channel = True):
        super().__init__(accross_channel = accross_channel)
        self.p = p
        self.q = q
        self.p_obs = p_obs
        self.cut = cut

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

        d_params = max(int(self.p_params * d), 1) if self.exclude_inputs else d ## number of variables used as inputs (at least 1)
        d_na = d - d_params if self.exclude_inputs else d ## number of variables masked with the logistic model


        mask = torch.zeros(batch_size, d).int()

        d_na = max(int(self.p_obs * d), 1) ## number of variables that will have NMAR values

        ### Sample variables that will have imps at the extremes
        idxs_na = np.random.choice(d, d_na, replace=False) ### select at least one variable with missing values

        ### check if values are greater/smaller that corresponding quantiles
        if self.cut == 'upper':
            quants = quantile(data_per_channel[:, idxs_na], 1-self.q, dim=0)
            m = data_per_channel[:, idxs_na] >= quants
        elif self.cut == 'lower':
            quants = quantile(data_per_channel[:, idxs_na], self.q, dim=0)
            m = data_per_channel[:, idxs_na] <= quants
        elif self.cut == 'both':
            u_quants = quantile(data_per_channel[:, idxs_na], 1-self.q, dim=0)
            l_quants = quantile(data_per_channel[:, idxs_na], self.q, dim=0)
            m = (data_per_channel[:, idxs_na] <= l_quants) | (data_per_channel[:, idxs_na] >= u_quants)

        ### Hide some values exceeding quantiles
        ber = torch.rand(batch_size, d_na)
        mask[:, idxs_na] = ((ber > self.p) & m).int()
        mask = mask.reshape(batch_size, *mask_size[1:]).expand(data.shape)

        return mask

 
