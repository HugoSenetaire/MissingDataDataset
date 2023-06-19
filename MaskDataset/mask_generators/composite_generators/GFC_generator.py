from .abstract_composite_image_generators import AbstractCompositeImageGenerator
from ..simple_generators.fixed_rectangle_image_generator import FixedRectangleGeneratorFromCoordinates
from ..simple_generators.mixture_mask_generator import MixtureMaskGenerator


class GFCGenerator(AbstractCompositeImageGenerator):
    """
    Generate equiprobably masks O1-O6 from the paper
    Li, Y., Liu, S., Yang, J., & Yang, M. H. Generative face completion.
    Conference on Computer Vision and Pattern Recognition, 2016.
    ArXiv link: https://arxiv.org/abs/1704.05838
    Note, that this generator works as supposed only for 128x128 images.
    """
    def __init__(self, shape_x = None, shape_y = None, accross_channel=True):
        super().__init__(shape_x= shape_x, shape_y = shape_y, accross_channel=accross_channel)
       
    def define_generator(self, batch):
        shape_x, shape_y = batch['data'].shape[2:]
        gfc_o1 = FixedRectangleGeneratorFromCoordinates(52/128.*shape_x, 33/128.*shape_y, 116/128.*shape_x, 71/128.*shape_y, accross_channel = self.accross_channel)
        gfc_o2 = FixedRectangleGeneratorFromCoordinates(52/128.*shape_x, 57/128.*shape_y, 116/128.*shape_x, 95/128.*shape_y, accross_channel = self.accross_channel)
        gfc_o3 = FixedRectangleGeneratorFromCoordinates(52/128.*shape_x, 29/128.*shape_y, 74/128.*shape_x, 99/128.*shape_y, accross_channel = self.accross_channel)
        gfc_o4 = FixedRectangleGeneratorFromCoordinates(52/128.*shape_x, 29/128.*shape_y, 74/128.*shape_x, 67/128.*shape_y, accross_channel = self.accross_channel)
        gfc_o5 = FixedRectangleGeneratorFromCoordinates(52/128.*shape_x, 61/128.*shape_y, 74/128.*shape_x, 99/128.*shape_y, accross_channel = self.accross_channel)
        gfc_o6 = FixedRectangleGeneratorFromCoordinates(86/128.*shape_x, 40/128.*shape_y, 124/128.*shape_x, 88/128.*shape_y, accross_channel = self.accross_channel)

        self.generator = MixtureMaskGenerator([
            gfc_o1, gfc_o2, gfc_o3, gfc_o4, gfc_o5, gfc_o6
        ], [1] * 6)