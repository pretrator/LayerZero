from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Any, Union
from .AugmentationMode import AugmentationMode
from .KorniaHelper import is_kornia_available, ensure_kornia


@dataclass
class ImageLoaderConfig:
    data_dir: str = "data"
    batch_size: int = 64
    channels: int = 3
    num_workers: Optional[int] = None
    shuffle_train: bool = True
    download: bool = True
    mean: Optional[Tuple[float, ...]] = None
    std: Optional[Tuple[float, ...]] = None
    persistent_workers: Optional[bool] = None
    prefetch_factor: int = 2
    augmentation_mode: AugmentationMode = AugmentationMode.BASIC
    use_gpu_augmentation: Any = 'auto'  # 'auto', True, False
    auto_install_kornia: bool = True
    extra_transforms: List[Callable] = field(default_factory=list)

class ImageDataLoader:
    def __init__(
        self,
        dataset_cls,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        config: Optional[ImageLoaderConfig] = None,
    ):
        if dataset_cls is None:
            raise ValueError("Dataset Class (dataset_cls) is not provided")
            
        self.image_size = self._validate_image_size(image_size) if image_size is not None else None
            
        self.dataset_cls = dataset_cls
        self.data_dir = (config.data_dir if config else "data")
        self.batch_size = (config.batch_size if config else 64)
        self.channels = (config.channels if config else 3)
        self.extra_transforms = (config.extra_transforms if config else [])
        
        # Auto-detect optimal num_workers based on device and CPU count
        cfg_num_workers = config.num_workers if config else None
        if cfg_num_workers is None:
            if torch.cuda.is_available():
                # For GPU: use more workers to keep GPU fed
                self.num_workers = min(4, torch.multiprocessing.cpu_count())
            else:
                # For CPU: fewer workers to avoid CPU contention
                # Data loading competes with model computation on CPU
                self.num_workers = min(2, max(1, torch.multiprocessing.cpu_count() // 2))
        else:
            self.num_workers = cfg_num_workers
            
        # persistent_workers reduces worker spawn overhead
        cfg_persistent_workers = config.persistent_workers if config else None
        if cfg_persistent_workers is None:
            self.persistent_workers = self.num_workers > 0
        else:
            self.persistent_workers = cfg_persistent_workers
            
        cfg_prefetch_factor = config.prefetch_factor if config else 2
        self.prefetch_factor = cfg_prefetch_factor if self.num_workers > 0 else None
        self.shuffle_train = (config.shuffle_train if config else True)
        self.download = (config.download if config else True)

        cfg_mean = config.mean if config else None
        cfg_std = config.std if config else None
        if cfg_mean is None or cfg_std is None:
            if self.channels == 1:
                self.mean, self.std = (0.5,), (0.5,)
            else:
                self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            self.mean, self.std = cfg_mean, cfg_std

        # Set augmentation mode
        cfg_aug_mode = config.augmentation_mode if config else AugmentationMode.BASIC
        if not isinstance(cfg_aug_mode, AugmentationMode):
            raise ValueError(
                f"augmentation_mode must be an AugmentationMode enum. "
                f"Got: {type(cfg_aug_mode)}. "
                f"Use: AugmentationMode.MINIMAL, .BASIC, or .STRONG"
            )
        self.augmentation_mode = cfg_aug_mode
        
        # Handle GPU augmentation (separate from augmentation intensity)
        cfg_use_gpu_aug = config.use_gpu_augmentation if config else 'auto'
        cfg_auto_install_kornia = config.auto_install_kornia if config else True
        if cfg_use_gpu_aug == 'auto':
            # Auto-detect: Use GPU if available and Kornia can be installed
            if torch.cuda.is_available():
                if not is_kornia_available() and cfg_auto_install_kornia:
                    print("\n" + "="*60)
                    print("🚀 GPU DETECTED! Setting up GPU-accelerated augmentation...")
                    print("="*60)
                    ensure_kornia(auto_install=True, verbose=True)
                
                self.use_gpu_augmentation = is_kornia_available() && config.augmentation_mode != AugmentationMode.OFF
                
                if self.use_gpu_augmentation:
                    print("\n" + "="*60)
                    print("⚡ GPU AUGMENTATION ENABLED ⚡")
                    print("="*60)
                    print(f"Mode: {self.augmentation_mode.name}")
                    print(f"Description: {self.augmentation_mode.description}")
                    print(f"Acceleration: Kornia GPU (5-10x faster than CPU)")
                    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'CUDA'}")
                    print("="*60 + "\n")
                else:
                    print(f"\nℹ️  Using {self.augmentation_mode.name} augmentations on CPU")
                    print("   Tip: Install Kornia for GPU acceleration: pip install kornia kornia-rs\n")
            else:
                # CPU only
                self.use_gpu_augmentation = False
                print(f"\nℹ️  CPU detected: Using {self.augmentation_mode.name} augmentations on CPU")
                print(f"   ({self.augmentation_mode.description})\n")
        else:
            # Explicit True/False
            self.use_gpu_augmentation = bool(cfg_use_gpu_aug)
            
            if self.use_gpu_augmentation:
                if not is_kornia_available():
                    if cfg_auto_install_kornia:
                        print("\n📦 GPU augmentation requested. Installing Kornia...")
                        ensure_kornia(auto_install=True, verbose=True)
                    
                    if not is_kornia_available():
                        print("\n⚠️  Warning: GPU augmentation requires Kornia but it's not available.")
                        print("   Falling back to CPU augmentation.")
                        print("   Install manually: pip install kornia kornia-rs\n")
                        self.use_gpu_augmentation = False
                
                if self.use_gpu_augmentation:
                    # Successfully enabled
                    print("\n" + "="*60)
                    print("⚡ GPU AUGMENTATION ENABLED ⚡")
                    print("="*60)
                    print(f"Mode: {self.augmentation_mode.name}")
                    print(f"Description: {self.augmentation_mode.description}")
                    print(f"Acceleration: Kornia GPU (5-10x faster than CPU)")
                    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'CUDA'}")
                    print("="*60 + "\n")
            else:
                # Explicitly disabled
                print(f"\nℹ️  GPU augmentation disabled. Using CPU augmentation.")
                print(f"   Mode: {self.augmentation_mode.name} ({self.augmentation_mode.description})\n")
        
        # All augmentation parameters are now handled directly in build_transforms

    def build_transforms(self, train: bool = True, extra_ops: Optional[List[Callable]] = None):
        """
        Build transform pipeline based on augmentation_mode.
        
        Augmentation intensity (CPU or GPU):
        - OFF: No augmentation (ToTensor, Normalize only)
        - MINIMAL: RandomCrop, RandomHorizontalFlip
        - BASIC: + ColorJitter (50% prob)
        - STRONG: + Rotation, RandomErasing
        """
        ops: List[Callable] = list(extra_ops) if extra_ops else []
        
        # Training mode transforms
        if train:
            # OFF mode - no augmentations
            if self.augmentation_mode == AugmentationMode.OFF:
                pass
            
            # MINIMAL mode
            elif self.augmentation_mode == AugmentationMode.MINIMAL:
                ops.append(transforms.RandomHorizontalFlip(p=0.5))
                if self.image_size:
                    resize_size = tuple(int(dim * 1.14) for dim in self.image_size)
                    ops.append(transforms.Resize(resize_size))
                    ops.append(transforms.RandomCrop(self.image_size))
            
            # BASIC mode
            elif self.augmentation_mode == AugmentationMode.BASIC:
                ops.append(transforms.RandomHorizontalFlip(p=0.5))
                ops.append(transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.5
                ))
                
                if self.image_size:
                    base_ratio = self.image_size[0] / self.image_size[1]
                    ratio = (base_ratio * 0.75, base_ratio * 1.33)
                    ops.append(transforms.RandomResizedCrop(
                        self.image_size,
                        scale=(0.2, 1.0),
                        ratio=ratio
                    ))
            
            # STRONG mode
            elif self.augmentation_mode == AugmentationMode.STRONG:
                ops.append(transforms.RandomHorizontalFlip(p=0.5))
                ops.append(transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8
                ))
                ops.append(transforms.RandomRotation(degrees=10))
                
                if self.image_size:
                    base_ratio = self.image_size[0] / self.image_size[1]
                    ratio = (base_ratio * 0.75, base_ratio * 1.33)
                    ops.append(transforms.RandomResizedCrop(
                        self.image_size,
                        scale=(0.08, 1.0),
                        ratio=ratio
                    ))
        
        # Always add ToTensor and Normalize
        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(self.mean, self.std))
        
        # Add RandomErasing for STRONG mode in training
        if train and self.augmentation_mode == AugmentationMode.STRONG:
            ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)))
            
        return transforms.Compose(ops)

    def get_gpu_augmentation(self, device='cuda'):
        """
        Create a GPUAugmentation instance matching the current augmentation mode.
        
        Only needed if use_gpu_augmentation=True and you want to apply
        augmentations manually in your training loop.
        
        Args:
            device (str): Device for GPU augmentation (default: 'cuda')
            
        Returns:
            GPUAugmentation instance or None if GPU augmentation not enabled
            
        Example:
            loader = ImageDataLoader(..., use_gpu_augmentation=True)
            train_loader, val_loader = loader.get_loaders()
            gpu_aug = loader.get_gpu_augmentation()
            
            for X, y in train_loader:
                X = X.to(device)
                X = gpu_aug(X)  # Apply GPU augmentation
                ...
        """
        if not self.use_gpu_augmentation:
            print(f"⚠️  GPU augmentation not enabled. Set use_gpu_augmentation=True or 'auto'")
            return None
        
        if not is_kornia_available():
            print("⚠️  Kornia not available. Install with: pip install kornia kornia-rs")
            return None
        
        try:
            from .GPUAugmentation import GPUAugmentation
            return GPUAugmentation(
                image_size=self.image_size,
                mode=self.augmentation_mode,  # Use same mode
                device=device,
                channels=self.channels  # Pass channels to avoid runtime detection
            )
        except ImportError as e:
            print(f"⚠️  Could not import GPUAugmentation: {e}")
            return None
    
    def get_loaders(self):
        """
        Create train and test DataLoaders.
        
        If augmentation_mode is GPU, you should also call get_gpu_augmentation()
        to apply GPU augmentations in your training loop.
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        train_dataset = self.dataset_cls(
            root=self.data_dir,
            train=True,
            download=self.download,
            transform=self.build_transforms(train=True, extra_ops=self.extra_transforms)
        )
        test_dataset = self.dataset_cls(
            root=self.data_dir,
            train=False,
            download=self.download,
            transform=self.build_transforms(train=False, extra_ops=self.extra_transforms)
        )

        # pin_memory speeds up CPU->GPU transfer but adds overhead on CPU-only
        use_pin_memory = torch.cuda.is_available()
        
        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': self.shuffle_train,
            'num_workers': self.num_workers,
            'pin_memory': use_pin_memory,
        }
        
        # Add persistent_workers and prefetch_factor only if num_workers > 0
        if self.num_workers > 0:
            train_loader_kwargs['persistent_workers'] = self.persistent_workers
            if self.prefetch_factor is not None:
                train_loader_kwargs['prefetch_factor'] = self.prefetch_factor
        
        train_loader = DataLoader(train_dataset, **train_loader_kwargs)
        # Attach ImageDataLoader reference for GPU augmentation auto-detection
        train_loader._image_data_loader = self

        test_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': use_pin_memory,
        }
        
        if self.num_workers > 0:
            test_loader_kwargs['persistent_workers'] = self.persistent_workers
            if self.prefetch_factor is not None:
                test_loader_kwargs['prefetch_factor'] = self.prefetch_factor
                
        test_loader = DataLoader(test_dataset, **test_loader_kwargs)
        # Attach ImageDataLoader reference for consistency
        test_loader._image_data_loader = self
        
        # Print usage instructions if GPU augmentation is enabled
        if self.use_gpu_augmentation:
            print("\n" + "="*60)
            print("💡 GPU AUGMENTATION USAGE")
            print("="*60)
            print("GPU augmentation requires manual application in your training loop:")
            print()
            print("  # Get GPU augmentation instance")
            print("  gpu_aug = loader.get_gpu_augmentation()")
            print()
            print("  # In training loop:")
            print("  for X, y in train_loader:")
            print("      X = X.to(device, non_blocking=True)")
            print("      X = gpu_aug(X)  # ← Apply GPU augmentation here")
            print("      ")
            print("      logits = model(X)")
            print("      loss = loss_fn(logits, y)")
            print("      ...")
            print("="*60 + "\n")

        return train_loader, test_loader

    @staticmethod
    def _validate_image_size(size: Optional[Union[int, Tuple[int, int], List[int]]]) -> Optional[Tuple[int, int]]:
        """
        Validate and normalize image size input.
        
        Args:
            size: Integer for square images or tuple/list of (height, width), or None to keep original size
            
        Returns:
            Optional[Tuple[int, int]]: Validated (height, width) or None if no resizing needed
            
        Raises:
            ValueError: If size is invalid
        """
        try:
            # Handle single integer (square)
            if isinstance(size, int):
                if size <= 0:
                    raise ValueError(f"Size must be positive, got {size}")
                return (size, size)
            
            # Handle sequence of 2 integers
            h, w = size  # Will raise ValueError if not sequence of 2
            if not (isinstance(h, int) and isinstance(w, int)):
                raise ValueError(f"Dimensions must be integers, got {type(h)}, {type(w)}")
            if h <= 0 or w <= 0:
                raise ValueError(f"Dimensions must be positive, got {h}, {w}")
            return (h, w)
            
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Image size must be positive int or (height, width) sequence, got {size}"
            ) from e

    @property
    def dimensions(self) -> dict:
        """
        Get the current image dimensions and aspect ratio information.
        
        Returns:
            dict: Contains:
                - height (Optional[int]): Image height if size is set, None otherwise
                - width (Optional[int]): Image width if size is set, None otherwise
                - aspect_ratio (Optional[float]): Width/Height ratio if size is set, None otherwise
                - is_square (Optional[bool]): Whether the image is square if size is set, None otherwise
                - is_original_size (bool): Whether original image size is preserved
        """
        if self.image_size is None:
            return {
                'height': None,
                'width': None,
                'aspect_ratio': None,
                'is_square': None,
                'is_original_size': True
            }
            
        height, width = self.image_size
        return {
            'height': height,
            'width': width,
            'aspect_ratio': width / height,
            'is_square': height == width,
            'is_original_size': False
        }
