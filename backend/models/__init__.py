"""
backend/models/__init__.py
Tea Leaf Disease Segmentation — Model Package
"""
from .unet    import SEAUNet, SEBlock, AttentionGate, ConvBlock, NUM_CLASSES
# These modules include plotting/augmentation libraries with native DLLs. Keep
# them optional so lightweight classifier training can start without importing
# the legacy segmentation visualisation stack.
try:
    from .metrics import (
        DiceLoss, CombinedDiceBCELoss, MultiClassCombinedLoss, TverskyLoss,
        calculate_severity, MetricsTracker, compute_all_metrics,
        calculate_operational_risk, visualise_metrics_curves,
        visualise_operational_risk,
    )
except (ImportError, OSError):
    pass

try:
    from .dataset import (
        TeaLeafSegmentationDataset, build_dataloaders, CLASS_NAMES,
        CLASS_TO_IDX, NUM_CLASSES as DATASET_NUM_CLASSES,
    )
except (ImportError, OSError):
    pass
from .plant import (
    PlantSegClassificationDataset,
    build_mobilenet_classifier,
    build_label_maps,
    load_plantseg_metadata,
)

__all__ = [
    "SEAUNet", "SEBlock", "AttentionGate", "ConvBlock", "NUM_CLASSES",
    "DiceLoss", "CombinedDiceBCELoss", "MultiClassCombinedLoss", "TverskyLoss",
    "calculate_severity",
    "MetricsTracker", "compute_all_metrics",
    "calculate_operational_risk",
    "visualise_metrics_curves", "visualise_operational_risk",
    "TeaLeafSegmentationDataset", "build_dataloaders",
    "CLASS_NAMES", "CLASS_TO_IDX",
    "PlantSegClassificationDataset", "build_mobilenet_classifier",
    "build_label_maps", "load_plantseg_metadata",
]
