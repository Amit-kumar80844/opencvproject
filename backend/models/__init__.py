"""
backend/models/__init__.py
Tea Leaf Disease Segmentation — Model Package
"""
from .unet    import SEAUNet, SEBlock, AttentionGate, ConvBlock, NUM_CLASSES
from .metrics import (
    DiceLoss,
    CombinedDiceBCELoss,
    MultiClassCombinedLoss,
    MetricsTracker,
    compute_all_metrics,
    calculate_operational_risk,
    visualise_metrics_curves,
    visualise_operational_risk,
)
from .dataset import (
    TeaLeafSegmentationDataset,
    build_dataloaders,
    CLASS_NAMES,
    CLASS_TO_IDX,
    NUM_CLASSES as DATASET_NUM_CLASSES,
)

__all__ = [
    "SEAUNet", "SEBlock", "AttentionGate", "ConvBlock", "NUM_CLASSES",
    "DiceLoss", "CombinedDiceBCELoss", "MultiClassCombinedLoss",
    "MetricsTracker", "compute_all_metrics",
    "calculate_operational_risk",
    "visualise_metrics_curves", "visualise_operational_risk",
    "TeaLeafSegmentationDataset", "build_dataloaders",
    "CLASS_NAMES", "CLASS_TO_IDX",
]
