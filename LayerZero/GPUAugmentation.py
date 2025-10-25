"""
GPU-Accelerated Augmentation using Kornia

This module provides GPU-accelerated image augmentations that are 5-10x faster
than CPU-based torchvision transforms. Augmentations run on batches after
data loading, utilizing GPU compute.

Installation:
    pip install kornia kornia-rs

Usage:
    from LayerZero import GPUAugmentation
    
    aug = GPUAugmentation(image_size=224, device='cuda')
    
    # In training loop:
    for X, y in dataloader:
        X = X.to(device)
        X = aug(X)  # Apply augmentations on GPU
        ...
"""

import torch
import torch.nn as nn
from .AugmentationMode import AugmentationMode
from .KorniaHelper import is_kornia_available, ensure_kornia


class GPUAugmentation(nn.Module):
    """
    GPU-accelerated augmentation pipeline using Kornia.
    
    Benefits:
    - 5-10x faster than CPU torchvision transforms
    - Operates on batches (more efficient than per-image)
    - Utilizes GPU compute (frees CPU for other tasks)
    - Fully differentiable (can be used in training)
    - Auto-detects grayscale vs RGB (skips color augs for grayscale)
    
    Args:
        width (int): Target width for the output images
        height (int): Target height for the output images
        mode (AugmentationMode): OFF, MINIMAL, BASIC, or STRONG
        device (str): 'cuda' or 'cpu'
        p (float): Probability of applying augmentations
        channels (int, optional): Number of channels (1=grayscale, 3=RGB). Auto-detected if None.
        
    Note:
        Both width and height must be positive integers.
        The dimensions should be provided by the ImageDataLoader or other calling code.
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        mode=AugmentationMode.BASIC,
        device='cuda',
        p=0.5,
        channels=None,  # Auto-detect if None
    ):
        super().__init__()
        
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("Width and height must be integers")
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
            
        self.width = width
        self.height = height
        self.mode = mode
        self.device = device
        self.channels = channels  # Will be auto-detected on first forward pass if None
        
        # Always try to install Kornia, even if mode is OFF (user might switch modes later)
        try:
            if not is_kornia_available():
                print("\n📦 Installing Kornia...")
                ensure_kornia(auto_install=True, verbose=True)
            
            # Import Kornia (will be used if mode is not OFF)
            import kornia.augmentation as K
            self.K = K
        except Exception as e:
            if mode != AugmentationMode.OFF:
                raise ImportError(
                    "Failed to install/import Kornia. Install manually with: pip install kornia kornia-rs"
                ) from e
            else:
                print(f"⚠️  Note: Kornia installation failed but augmentation mode is OFF, continuing without it")
                self.K = None
        
        self.transforms = None  # Will be initialized on first forward pass
        self._initialized = False
        
    def _build_transforms(self, channels):
        """Build augmentation pipeline based on number of channels."""
        augs = []
        is_grayscale = (channels == 1)
        
        if self.mode == AugmentationMode.OFF:
            # No augmentation
            augs = []
            
        elif self.mode == AugmentationMode.MINIMAL:
            # MINIMAL: Fast augmentations only (geometry only, no color)
            augs = [
                self.K.RandomHorizontalFlip(p=0.5),
                self.K.RandomCrop((self.height, self.width), pad_if_needed=True),
            ]
            
        elif self.mode == AugmentationMode.BASIC:
            # BASIC: Standard augmentations
            augs = [
                self.K.RandomResizedCrop((self.height, self.width), scale=(0.2, 1.0), ratio=(0.75, 1.33), p=1.0),
                self.K.RandomHorizontalFlip(p=0.5),
            ]
            # Only add color augmentations for RGB images
            if not is_grayscale:
                augs.append(self.K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.5))
            
        elif self.mode == AugmentationMode.STRONG:
            # STRONG: Maximum augmentation strength
            augs = [
                self.K.RandomResizedCrop((self.height, self.width), scale=(0.08, 1.0), ratio=(0.75, 1.33), p=1.0),
                self.K.RandomHorizontalFlip(p=0.5),
            ]
            # Only add color augmentations for RGB images
            if not is_grayscale:
                augs.append(self.K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8))
            # Geometry augmentations work for both RGB and grayscale
            augs.extend([
                self.K.RandomRotation(degrees=10.0, p=0.3),
                self.K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.2),
                self.K.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            ])
        
        # Create augmentation container
        transforms = self.K.AugmentationSequential(*augs, data_keys=["input"])
        return transforms.to(self.device)
    
    def forward(self, x):
        """
        Apply augmentations to a batch of images.
        
        Args:
            x (torch.Tensor): Batch of images [B, C, H, W]
            
        Returns:
            torch.Tensor: Augmented images [B, C, H, W]
        """
        # When mode is OFF, bypass everything and return input as is
        if self.mode == AugmentationMode.OFF:
            return x
            
        # Auto-detect channels on first forward pass
        if not self._initialized:
            if self.channels is None:
                self.channels = x.shape[1]  # Detect from input
            self.transforms = self._build_transforms(self.channels)
            self._initialized = True
            
            # Log what's being used
            aug_type = "Grayscale" if self.channels == 1 else "RGB"
            print(f"🎨 GPU Aug: {aug_type} ({self.mode.name}) | {len(self.transforms)} transforms")
        
        # Kornia expects input in range [0, 1]
        return self.transforms(x)
    
    def __repr__(self):
        num_transforms = len(self.transforms) if self._initialized else "auto"
        return f"GPUAugmentation(mode={self.mode}, device={self.device}, transforms={num_transforms})"


class HybridAugmentation(nn.Module):
    """
    Hybrid augmentation: Light CPU transforms + Heavy GPU transforms
    
    This is the optimal strategy:
    1. CPU: Only essential transforms (ToTensor, Normalize)
    2. GPU: All heavy augmentations on batched data
    
    Result: Best performance, especially on multi-worker DataLoaders
    """
    
    def __init__(self, width: int, height: int, mode='standard', device='cuda'):
        super().__init__()
        self.gpu_aug = GPUAugmentation(width=width, height=height, mode=mode, device=device)
    
    def forward(self, x):
        """Apply GPU augmentations to batch"""
        return self.gpu_aug(x)


# Example integration with Trainer
class AugmentedTrainingLoop:
    """
    Example of how to integrate GPU augmentations into training.
    
    Usage:
        gpu_aug = GPUAugmentation(width=224, height=224, mode=AugmentationMode.BASIC, device='cuda')
        
        for X, y in dataloader:
            X = X.to(device)
            X = gpu_aug(X)  # Apply GPU augmentations
            
            # Continue with training...
            logits = model(X)
            loss = loss_fn(logits, y)
            ...
    """
    pass


if __name__ == "__main__":
    print("="*60)
    print("GPU Augmentation Benchmark")
    print("="*60)
    
    if is_kornia_available():
        results = benchmark_augmentation_speed()
        
        if 'speedup' in results:
            print(f"\n✅ GPU augmentation is {results['speedup']:.2f}x faster!")
            print(f"\nRecommendation: Use GPUAugmentation for 5-10x speedup")
        else:
            print(f"\n⚠️  GPU not available. GPU augmentation requires CUDA.")
    else:
        print("\n❌ Kornia not installed.")
        print("Install with: pip install kornia kornia-rs")

