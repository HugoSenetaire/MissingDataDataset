from .dual_mask_generator import DUAL_mask, DUAL_mask_opposite
from .fixed_rectangle_image_generator import FixedRectangleGeneratorFromCoordinates
from .MAR_MIWAE_mask_generator import MARMIWAEMaskGenerator
from .mask_target_correlated import MaskTargetCorrelated
from .MCAR_halfcut_generator import MCARHalfcut
from .MCAR_mask_generator import MCARGenerator
from .mixture_mask_generator import MixtureMaskGenerator
from .MNAR_quantile import MNARQuantile
from .monotonic_MCAR_mask_generator import MonotonicMCARMaskGenerator
from .random_pattern_mask_generator import RandomPattern
from .rectangle_image_generator import RectangleGenerator

dic_simple_generators = {
    "rectangle": RectangleGenerator,
    "fixed_rectangle": FixedRectangleGeneratorFromCoordinates,
    "MCAR": MCARGenerator,
    "MCAR_halfcut": MCARHalfcut,
    "MNAR_quantile": MNARQuantile,
    "MAR_MIWAE": MARMIWAEMaskGenerator,
    "monotonic_MCAR": MonotonicMCARMaskGenerator,
    "random_pattern": RandomPattern,
    "mixture": MixtureMaskGenerator,
    "mask_target_correlated": MaskTargetCorrelated,
    "DUAL_mask": DUAL_mask,
    "DUAL_mask_opposite": DUAL_mask_opposite,
}