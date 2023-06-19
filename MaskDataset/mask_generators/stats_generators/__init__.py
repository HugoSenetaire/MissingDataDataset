from .MAR_mask_generator import MARMaskGenerator
from .MNAR_mask_logistics import MNARMaskLogistics,MNARMaskSelfLogistics

dic_stats_generator = {
    "MAR": MARMaskGenerator,
    "MNAR_logistics": MNARMaskLogistics,
    "MNAR_self_logistics": MNARMaskSelfLogistics,
}
