
from ..abstract_mask_generator import AbstractGenerator

import numpy as np
import torch 

class MixtureMaskGenerator(AbstractGenerator):
    """
    For each object firstly sample a generator according to their weights,
    and then sample a mask from the sampled generator.
    """
    def __init__(self, generators, weights):
        self.generators = generators
        self.weights = weights

    def masking_rule(self, batch):
        w = np.array(self.weights, dtype='float')
        w /= w.sum()
        c_ids = np.random.choice(w.size, batch['data'].shape[0], True, w)
        mask = torch.zeros_like(batch['data'], device=batch['data'].device)
        for i, gen in enumerate(self.generators):
            ids = np.where(c_ids == i)[0]
            if len(ids) == 0:
                continue
            samples = gen.masking_rule(batch['data'][ids]).to(batch['data'].device)
            mask[ids] = samples
        return mask

