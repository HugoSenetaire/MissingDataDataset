

from ..abstract_mask_generator import AbstractGenerator

import torch
import numpy as np

class MCARHalfcut(AbstractGenerator):
    """
    Delete p_obs of the image with a probability p.
    The cut is made horizontally (rows) or vertically (columns) depending on the orientation parameter.

    Attributes:
    ----------
        p_obs (float): proportion of variable with no missing values that will define the rectangle
        p (float): probability of component to be observed among the variables with missing values
        orientation (str): 'rows' or 'columns'
        
    """
    def __init__(self, p, p_obs=0.5, orientation='rows', origin = 'start', accross_channel = True,):
        super().__init__(accross_channel = accross_channel)
        self.p = p
        self.p_obs = p_obs
        self.orientation = orientation
        self.origin = origin


 

    def masking_rule(self, batch):
        assert len(batch['data'].shape) == 4, "MCARHalfcut only supports 4D tensors, ie images with channel included"
        if self.orientation == 'columns':
            batch['data'] = batch['data'].permute(0,1,3,2)

        if self.accross_channel:
            size_mask = (batch['data'].shape[0], 1, *batch['data'].shape[2:])
        else :
            size_mask = batch['data'].shape


        self.rectangle_size = int((1-self.p_obs)*size_mask[1])
 
        bernoulli_mask = torch.ones(size_mask)
        ber = torch.rand(size_mask[0], size_mask[1],).unsqueeze(-1).unsqueeze(-1).expand(size_mask)
        if self.origin == 'start':
            bernoulli_mask[:, :, self.rectangle_size:, :] = (ber[:, :, self.rectangle_size:, :] > self.p).int()
        elif self.origin == 'end':
            bernoulli_mask[:, :, :self.rectangle_size, :] = (ber[:, :, :self.rectangle_size, :] > self.p).int()
        else:
            raise ValueError("origin should be 'start' or 'end'")

        if self.orientation == 'columns':
            bernoulli_mask = bernoulli_mask.permute(0,1,3,2)
            batch['data'] = batch['data'].permute(0,1,3,2)

        return bernoulli_mask
    
