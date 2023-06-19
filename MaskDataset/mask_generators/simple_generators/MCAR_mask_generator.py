
from ..abstract_mask_generator import AbstractGenerator

import torch
import numpy as np

class MCARGenerator(AbstractGenerator):
    """
    Returned mask is sampled from component-wise independent Bernoulli
    distribution with probability of component to be unobserved p among 1-p_obs variables with missing values.
    Such mask induces the type of missingness which is called
    in literature "missing completely at random" (MCAR).
    If some value in batch is missed, it automatically becomes unobserved.

    Attributes:
    ----------
        p_obs (float): proportion of variable with no missing values
        p (float): probability of component to be observed among the variables with missing values
        mask (torch.Tensor): mask for batch 
        
    """
    def __init__(self, p, p_obs=0.0, accross_channel = True):
        super().__init__(accross_channel = accross_channel)
        self.p = p
        self.p_obs = p_obs
        self.idx_obs = None
        self.idx_na = None

 

    def masking_rule(self, batch):
        if self.accross_channel:
            size_mask = (batch['data'].shape[0], 1, *batch['data'].shape[2:])
        else :
            size_mask = batch['data'].shape

        total_dim = np.prod(size_mask[1:]).astype(int)

        if self.idx_na is None or self.idx_obs is None:
            d_obs = int(self.p_obs * total_dim) ## number of variables that will have no missing values (at least one variable)
            d_na = total_dim - d_obs ## number of variables that will have missing values
            self.idxs_obs = np.random.choice(total_dim, d_obs, replace=False)
            self.idxs_nas = np.array([i for i in range(total_dim) if i not in self.idxs_obs])

        bernoulli_mask = torch.ones(size_mask).flatten(1)
        bernoulli_mask[:, self.idxs_nas] = (torch.rand(size_mask[0], len(self.idxs_nas)) > self.p).int()
        bernoulli_mask = bernoulli_mask.reshape(size_mask)
        return bernoulli_mask
    
