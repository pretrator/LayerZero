# LayerZero/__init__.py
__version__ = "0.3.0"

from .ImageDataLoader import ImageDataLoader
from .Helper import Helper
from .Trainer import (
    Trainer, 
    TrainerConfig, 
    Callback, 
    EarlyStopping, 
    CheckpointCallback, 
    TensorBoardCallback
)
from .AugmentationMode import AugmentationMode

# Kornia helper utilities (always available)
from .KorniaHelper import (
    KorniaHelper,
    is_kornia_available,
    install_kornia,
    ensure_kornia,
    get_kornia_version,
)

# Optional GPU augmentation (requires kornia)
try:
    from .GPUAugmentation import GPUAugmentation, HybridAugmentation
    __all__ = [
        "ImageDataLoader", 
        "Helper", 
        "Trainer", 
        "TrainerConfig",
        "Callback",
        "EarlyStopping",
        "CheckpointCallback",
        "TensorBoardCallback",
        "AugmentationMode",
        "GPUAugmentation", 
        "HybridAugmentation",
        "KorniaHelper",
        "is_kornia_available",
        "install_kornia",
        "ensure_kornia",
        "get_kornia_version",
    ]
except ImportError:
    __all__ = [
        "ImageDataLoader", 
        "Helper", 
        "Trainer", 
        "TrainerConfig",
        "Callback",
        "EarlyStopping",
        "CheckpointCallback",
        "TensorBoardCallback",
        "AugmentationMode",
        "KorniaHelper",
        "is_kornia_available",
        "install_kornia",
        "ensure_kornia",
        "get_kornia_version",
    ]
