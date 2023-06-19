from .abstract_composite_image_generators import AbstractCompositeImageGenerator
from ..simple_generators.fixed_rectangle_image_generator import FixedRectangleGeneratorFromCoordinates
from ..simple_generators.MCAR_mask_generator import MCARGenerator
from ..simple_generators.mixture_mask_generator import MixtureMaskGenerator
from ..simple_generators.random_pattern_mask_generator import RandomPattern

import torch

class SIIDGMGenerator(AbstractCompositeImageGenerator):
    """
    Generate equiprobably masks from the paper
    Yeh, R. A., Chen, C., Yian Lim, T., Schwing,
    A. G., Hasegawa-Johnson, M., & Do, M. N.
    Semantic Image Inpainting with Deep Generative Models.
    Conference on Computer Vision and Pattern Recognition, 2017.
    ArXiv link: https://arxiv.org/abs/1607.07539
    Note, that this generator works as supposed only for 128x128 images.
    In the paper authors used 64x64 images, but here for the demonstration
    purposes we adapted their masks to 128x128 images.
    """
    def __init__(self, shape_x=None, shape_y = None, accross_channel=True):
        super().__init__(shape_x=shape_x, shape_y = shape_y, accross_channel=accross_channel)
       

    def define_generators(self, batch):
        self.shape_x, self.shape_y = batch['data'].shape[2:]
        # the resolution parameter differs from the original paper because of
        # the image size change from 64x64 to 128x128 in order to preserve
        # the typical mask shapes
        random_pattern = RandomPattern(max_size=10000, resolution=0.03)
        # the number of missing pixels is also increased from 80% to 95%
        # in order not to increase the amount of the observable information
        # for the inpainting method with respect to the original paper
        # with 64x64 images
        mcar = MCARGenerator(0.95)
        center = FixedRectangleGeneratorFromCoordinates(0.25 * self.shape_x, 0.25 * self.shape_y, 0.75 * self.shape_x, 0.75 * self.shape_y)    
        half_01 = FixedRectangleGeneratorFromCoordinates(0, 0, self.shape_x, 0.5 * self.shape_y)
        half_02 = FixedRectangleGeneratorFromCoordinates(0, 0, 0.5 * self.shape_x, self.shape_y)
        half_03 = FixedRectangleGeneratorFromCoordinates(0, 0.5*self.shape_y, self.shape_x, self.shape_y)
        half_04 = FixedRectangleGeneratorFromCoordinates(0.5*self.shape_x, 0, self.shape_x, self.shape_y)

        self.generator = MixtureMaskGenerator([
            center, random_pattern, mcar, half_01, half_02, half_03, half_04
        ], [2, 2, 2, 1, 1, 1, 1])

