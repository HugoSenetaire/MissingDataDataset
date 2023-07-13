from .cell_count import CellCount
from .d1regression_1 import d1Regression_1
from .d1regression_2 import d1Regression_2
from .head_pose_biwi import HeadPoseBIWI
from .steering_angle import SteeringAngle
from .UTKFaceDataset import UTKFace

EBM_FOR_REGRESSION_DATASETS = {
    "CellCount": CellCount,
    "1d_regression_1": d1Regression_1,
    "1d_regression_2": d1Regression_2,
    "HeadPoseBIWI": HeadPoseBIWI,
    "SteeringAngle": SteeringAngle,
    "UTKFace": UTKFace,
}