from ..abstract_mask_generator import AbstractGenerator
import torch
import numpy as np

class DUAL_mask(AbstractGenerator):
    """
    Missing mechanism with two types of mask. One where everything is observed and second one
    where we miss p_obs percent of the variables. p corresponds to the number of items with no missing values.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p : float
        Proportion of data points with no missing values.
    p_obs : float
        Proportion of observed values for data points which will have missing values.
    to_del : str, default = "last"
        If "last", the last variables will be deleted. If "first", the first variables will be deleted. 
        If "random", a random subset of variables will be deleted.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    def __init__(self, p=0.5, p_obs = 0.20, place = "last", accross_channel = True):
        super().__init__(accross_channel = accross_channel)
        self.p = p
        self.p_obs = p_obs
        self.place = place
        self.patterns = None
        self.idx_nas = None
        self.idx_obs = None

    def get_pattern(self, data):
        if self.accross_channel:
            n, c, dim = data.shape[0], 1, data.shape[2:]
        else:
            n, c, dim = data.shape[0], data.shape[1], data.shape[2:]

        self.c = c
        self.dim = dim
        d = np.prod(dim).astype(int) * c

        d_obs = max(int(self.p_obs * d),1) ## number of observed variables for data points with missing values
        d_na = d - d_obs ## number of missing variables for data points with missing values
    

    ### Sample variables that will all be observed, and those with missing values:
        if self.place == "random":
            idxs_obs = np.random.choice(d, d_obs, replace=False)
            idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])
        elif self.place == "first":
            idxs_obs = np.arange(d - d_obs, d)
            idxs_nas = np.arange(d - d_obs)
        elif self.place == "last":
            idxs_obs = np.arange(d_obs)
            idxs_nas = np.arange(d_obs, d)

        self.idx_obs = idxs_obs
        self.idx_nas = idxs_nas
        self.patterns = [np.ones(d), np.ones(d)]
        self.patterns[1][idxs_nas] = 0
        self.patterns[0][idxs_obs] = 0


    def masking_rule(self, batch):
        data = batch['data']
        n = data.shape[0]

        if self.patterns is None:
            self.get_pattern(data)

        mask = torch.ones((n, self.c, *self.dim)).flatten(1)
        ber = torch.rand(n).reshape(n, 1).expand(n, self.d_na)
        mask[:, self.idx_nas] = ber < self.p 

        mask = mask.reshape(n, self.c, *self.dim)

        return mask


class DUAL_mask_opposite(DUAL_mask):
    """
    Missing mechanism with two types of mask. One where p_obs features are observed and second one
    where the other are observed. p corresponds to the number of items with the first method.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p_obs : float
        Proportion of observed values for the first pattern.
    p : float
        Proportion of data points with the first pattern
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    def __init__(self, p=0.5, p_obs = 0.20, place = "last", accross_channel = True):
        super().__init__(p=p, p_obs = p_obs, place = place, accross_channel = accross_channel)

    def masking_rule(self, batch):
        data = batch['data']
        n = data.shape[0]

        if self.patterns is None:
            self.get_pattern(data)

        mask = torch.ones((n, self.c, *self.dim)).flatten(1)
        ber = torch.rand(n).reshape(n, 1).expand(n, self.d_na)
        mask[:, self.idx_nas] = (ber < self.p).int() 
        mask[:, self.idx_obs] = (ber > self.p).int()

        mask = mask.reshape(n, self.c, *self.dim).expand(data.shape)

        return mask

        

