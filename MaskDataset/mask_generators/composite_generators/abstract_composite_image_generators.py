from ..abstract_mask_generator import AbstractGenerator

import torch


class AbstractCompositeImageGenerator(AbstractGenerator):
    def __init__(self, shape_x = None, shape_y=None, accross_channel=True):
        super().__init__(accross_channel)
        self.shape_x = shape_x
        self.shape_y = shape_y
        if self.shape_x is None or self.shape_y is None:
            self.defined = False
            self.generator = None
        else :
            self.defined = True
            self.generator = self.define_generator(torch.zeros((1, 1, self.shape_x, self.shape_y)))

    def define_generator(self, batch):
        raise NotImplementedError("Please Implement this method")
    
    def masking_rule(self, batch):
        if not self.defined:
            self.define_generator(batch)
            self.defined = True
            
        assert self.shape_x == batch['data'].shape[2] and self.shape_y == batch['data'].shape[3], "The shape of the data is not the same as the one used to define the mask generator"
        return self.generator.masking_rule(batch)