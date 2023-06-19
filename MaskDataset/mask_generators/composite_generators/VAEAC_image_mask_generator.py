
from .abstract_composite_image_generators import AbstractCompositeImageGenerator
from .GFC_generator import GFCGenerator
from .SIIDGM_generator import SIIDGMGenerator
from ..simple_generators.rectangle_image_generator import RectangleGenerator
from ..simple_generators.mixture_mask_generator import MixtureMaskGenerator


class VAEACImageMaskGenerator(AbstractCompositeImageGenerator):
    """
    In order to train one model for the masks from all papers
    we mention above together with arbitrary rectangle masks,
    we use the mixture of all these masks at the training stage
    and on the test stage.
    Note, that this generator works as supposed only for 128x128 images.
    """
    def __init__(self, shape_x=None, shape_y=None, accross_channel=True):
        super().__init__(shape_x, shape_y, accross_channel=accross_channel)
       
    def define_generators(self, batch):
        self.shape_x, self.shape_y = batch['data'].shape[2:]
        siidgm = SIIDGMGenerator(self.shape_x, self.shape_y)
        gfc = GFCGenerator(self.shape_x, self.shape_y)
        common = RectangleGenerator()
        self.generator = MixtureMaskGenerator([siidgm, gfc, common], [1, 1, 2])
