from ..abstract_mask_generator import AbstractGenerator
import torch
import numpy as np

class MonotonicMCARMaskGenerator(AbstractGenerator):
    '''
        Missing completely at random mechanism where mask are included into each other.
        Intended for 1D data.
    '''

    def __init__(self, p_obs = 0.20, place = "last", accross_channel = True):
        super().__init__(accross_channel = accross_channel)
        self.p_obs = p_obs
        self.place = place

    def masking_rule(self, batch):
        data = batch['data']

        if self.accross_channel:
            size_mask = (data.shape[0], 1, *data.shape[2:])
        else :
            size_mask = data.shape
       
        dim = np.prod(size_mask[1:]).astype(int)
        batch_size = size_mask[0]
        mask = torch.zeros(size_mask, device = data.device).flatten(1)
        max_missing = int((1-self.p_obs) * dim)

        unif = torch.rand(batch_size,).unsqueeze(-1).expand(batch_size, max_missing)
        if self.place == "last":
            int_place = torch.arange(1-1./(max_missing+1), 1./(max_missing+1)-1e-8, step = -1./(max_missing+1)).unsqueeze(0).expand(batch_size, max_missing)
            mask[:, :, -max_missing:] = (unif > int_place).int()
        elif self.place == "first":
            int_place = torch.arange(1./(max_missing+1), 1-1./(max_missing+1)+1e-8, step = 1./(max_missing+1)).unsqueeze(0).expand(batch_size, max_missing)
            mask[:, :, :max_missing] = (unif > int_place).int()
        else:
            raise ValueError("place should be either 'last' or 'first'")
        
        mask = mask.reshape(size_mask).expand(data.shape)

        return mask

