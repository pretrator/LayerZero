"""
Augmentation mode enums for LayerZero
"""

from enum import Enum, auto


class AugmentationMode(Enum):
    """
    Augmentation intensity modes for ImageDataLoader.
    
    Clear, intent-based modes focusing on the speed/quality tradeoff:
    
    Modes:
        OFF: No augmentation (fastest, for debugging/testing)
        MINIMAL: Basic augmentations only (RandomCrop, Flip)
                 → Best for: Fast iteration, debugging, CPU systems
                 → Speed: Fastest (baseline)
                 → Quality: -2 to -4% accuracy
        
        BASIC: Standard augmentations (Crop, Flip, ColorJitter)
               → Best for: Production, balanced workflows
               → Speed: 2-3x slower than MINIMAL
               → Quality: -0.5 to -1% accuracy
        
        STRONG: Heavy augmentations (Crop, Flip, ColorJitter, Rotation, RandAugment)
                → Best for: Research, maximum accuracy
                → Speed: 5-8x slower than MINIMAL
                → Quality: Best (baseline)
    
    Note: GPU acceleration (Kornia) is controlled separately via use_gpu_augmentation
          and applies automatically when available to any mode (2-10x speed boost).
    """
    OFF = auto()       # No augmentation
    MINIMAL = auto()   # Fast: RandomCrop, RandomHorizontalFlip only
    BASIC = auto()     # Balanced: + ColorJitter (reduced probability)
    STRONG = auto()    # Quality: + Rotation, RandAugment/TrivialAugment, RandomErasing
    
    def __str__(self):
        return self.name.lower()
    
    @property
    def description(self):
        """Get human-readable description"""
        descriptions = {
            AugmentationMode.OFF: "No augmentation (debugging/testing)",
            AugmentationMode.MINIMAL: "Minimal augmentations (fast iteration, CPU)",
            AugmentationMode.BASIC: "Basic augmentations (production, balanced)",
            AugmentationMode.STRONG: "Strong augmentations (research, max quality)",
        }
        return descriptions.get(self, "Unknown mode")

