from ..abstract_mask_generator import AbstractGenerator

import torch
import numpy as np


class MARMIWAEMaskGenerator(AbstractGenerator):

    """
    Used for MNIST in MIWAE (https://proceedings.mlr.press/v97/mattei19a.html),
    We consider a MAR version of MNIST where all bottom
    halves of the pixels are observed. For each digit, either
    the top half, top quarter, or second quarter, is missing
    (depending on the number of white pixels in the bottom half).

    Attributes:
    ----------
        orientation (str): 'rows' or 'columns'
    """
    def __init__(self, orientation='rows', accross_channel = True):
        super().__init__(accross_channel = accross_channel)
        self.orientation = orientation
 

    def masking_rule(self, batch):
        X = batch['data']
        if self.orientation == 'columns':
            X = X.permute(0,1,3,2)

        n, c, dim = X.shape[0], X.shape[1], X.shape[2:]
        total_dim = np.prod(dim).astype(int)
       
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
        mask[:,:, idx_half:, :] = 0
        mask[:,:, :idx_quarter,] = (h[:,:, :idx_quarter,] != 0).int()
        mask[:,:, idx_quarter:idx_half,] = (h[:,:, idx_quarter:idx_half,] != 2).int()

        if self.orientation == 'columns':
            mask = mask.permute(0,1,3,2)
            X = X.permute(0,1,3,2)

        return mask
