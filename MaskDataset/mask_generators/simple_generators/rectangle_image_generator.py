

from ..abstract_mask_generator import AbstractGenerator

import torch
import numpy as np


class RectangleGenerator(AbstractGenerator):
    """
    Generates for each object a mask where unobserved region is
    a rectangle which square divided by the image square is in
    interval [min_rect_rel_square, max_rect_rel_square].
    Rectangle is created anew for each object.
    """
    def __init__(self, min_rect_rel_square=0.3, max_rect_rel_square=1, accross_channel=True):
        super().__init__(accross_channel=accross_channel)
        self.min_rect_rel_square = min_rect_rel_square
        self.max_rect_rel_square = max_rect_rel_square

    def gen_coordinates(self, width, height):
        x1, x2 = np.random.randint(0, width, 2)
        y1, y2 = np.random.randint(0, height, 2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return int(x1), int(y1), int(x2), int(y2)

    def masking_rule(self, batch):
        assert len(batch['data'].shape) == 4, "RectangleGenerator only supports 4D tensors, ie images with channel included"
        batch_size, num_channels, width, height = batch['data'].shape
        mask = torch.ones_like(batch['data'])
        for i in range(batch_size):
            x1, y1, x2, y2 = self.gen_coordinates(width, height)
            sqr = width * height
            while not (self.min_rect_rel_square * sqr <=
                       (x2 - x1 + 1) * (y2 - y1 + 1) <=
                       self.max_rect_rel_square * sqr):
                x1, y1, x2, y2 = self.gen_coordinates(width, height)
            mask[i, :, x1: x2 + 1, y1: y2 + 1] = 0
        return mask
