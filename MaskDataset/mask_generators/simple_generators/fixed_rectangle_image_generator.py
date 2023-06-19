
from ..abstract_mask_generator import AbstractGenerator
import torch
import numpy as np

class FixedRectangleGeneratorFromCoordinates(AbstractGenerator):
    """
    Always return an inpainting mask where unobserved region is
    a rectangle with corners in (x1, y1) and (x2, y2).
    """
    def __init__(self, x1, y1, x2, y2, accross_channel = True):
        super().__init__(accross_channel = accross_channel)
        if not self.accross_channel:
            raise AttributeError("FixedRectangleGenerator only supports accross_channel=True")
        self.x1 = int(x1)   
        self.x2 = int(x2)
        self.y1 = int(y1)
        self.y2 = int(y2)


    def masking_rule(self, batch):
        assert len(batch['data'].shape) == 4, "FixedRectangleGenerator only supports 4D tensors, ie images with channel included"
        mask = torch.ones_like(batch['data'])
        mask[:, :, self.x1: self.x2, self.y1: self.y2] = 0
        return mask

