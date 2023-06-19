

from ..abstract_mask_generator import AbstractGenerator

import torch
import numpy as np


class MaskTargetCorrelated(AbstractGenerator):
    """
    Each target has an associated masking rate.
    """
    def __init__(self, dic=None, accross_channel = True):
        super().__init__(accross_channel = accross_channel)
        self.dic = dic

    def masking_rule(self, batch):
        data = batch['data']
        target = batch['target']
        if self.dic is None :
            nb_targets = len(np.unique(target))
            rates = np.linspace(0.1, 0.9, nb_targets)
            self.dic = {i:rates[i] for i in range(nb_targets)}
        
        n, c, dim = data.shape[0], data.shape[1], data.shape[2:]
        if self.accross_channel:
            mask = torch.ones(n, 1, *dim)
        else :
            mask = torch.ones(n, c, *dim)

        p = torch.tensor([self.dic[int(y)] for y in target], device = data.device).reshape(-1, 1, *[1]*len(dim)).expand(n, 1, *dim)
        ber = torch.rand(n, 1, *dim)
        mask = (ber>p).int()

        return mask
