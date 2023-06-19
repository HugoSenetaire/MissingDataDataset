from .GFC_generator import GFCGenerator
from .SIIDGM_generator import SIIDGMGenerator
from .VAEAC_image_mask_generator import VAEACImageMaskGenerator

dic_composite_generators = {
    "GFC": GFCGenerator,
    "SIIDGM": SIIDGMGenerator,
    "VAEAC": VAEACImageMaskGenerator,
}