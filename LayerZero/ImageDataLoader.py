from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from .AugmentationMode import AugmentationMode
from .KorniaHelper import is_kornia_available, ensure_kornia

class ImageDataLoader:
    def __init__(
        self,
        dataset_cls,
        image_size,
        data_dir="data",
        batch_size=64,
        channels=3,
        num_workers=None,  # Auto-detect optimal value
        shuffle_train=True,
        download=True,
        mean=None,
        std=None,
        use_trivialaugment=True,
        use_randaugment=False,
        rand_n=2,
        rand_m=9,
        color_jitter=(0.4, 0.4, 0.4, 0.1),
        random_erase_p=0.25,
        persistent_workers=None,  # Keep workers alive between epochs
        prefetch_factor=2,  # Number of batches to prefetch per worker
        augmentation_mode=AugmentationMode.BASIC,  # AugmentationMode.OFF, .MINIMAL, .BASIC, .STRONG
        use_gpu_augmentation='auto',  # 'auto', True, False - GPU acceleration via Kornia
        auto_install_kornia=True,  # Auto-install Kornia for GPU augmentation
    ):
        if(dataset_cls is None):
            raise Exception("Dataset Class (dataset_cls) is not Provided")
        if(image_size is None):
            raise Exception("Image Size(image_size) is not provided")
        self.dataset_cls = dataset_cls
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        
        # Auto-detect optimal num_workers based on device and CPU count
        if num_workers is None:
            if torch.cuda.is_available():
                # For GPU: use more workers to keep GPU fed
                self.num_workers = min(4, torch.multiprocessing.cpu_count())
            else:
                # For CPU: fewer workers to avoid CPU contention
                # Data loading competes with model computation on CPU
                self.num_workers = min(2, max(1, torch.multiprocessing.cpu_count() // 2))
        else:
            self.num_workers = num_workers
            
        # persistent_workers reduces worker spawn overhead
        if persistent_workers is None:
            self.persistent_workers = self.num_workers > 0
        else:
            self.persistent_workers = persistent_workers
            
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None
        self.shuffle_train = shuffle_train
        self.download = download

        if mean is None or std is None:
            if channels == 1:
                self.mean, self.std = (0.5,), (0.5,)
            else:
                self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            self.mean, self.std = mean, std

        # Set augmentation mode
        if not isinstance(augmentation_mode, AugmentationMode):
            raise ValueError(
                f"augmentation_mode must be an AugmentationMode enum. "
                f"Got: {type(augmentation_mode)}. "
                f"Use: AugmentationMode.MINIMAL, .BASIC, or .STRONG"
            )
        self.augmentation_mode = augmentation_mode
        
        # Handle GPU augmentation (separate from augmentation intensity)
        if use_gpu_augmentation == 'auto':
            # Auto-detect: Use GPU if available and Kornia can be installed
            if torch.cuda.is_available():
                if not is_kornia_available() and auto_install_kornia:
                    print("\n" + "="*60)
                    print("🚀 GPU DETECTED! Setting up GPU-accelerated augmentation...")
                    print("="*60)
                    ensure_kornia(auto_install=True, verbose=True)
                
                self.use_gpu_augmentation = is_kornia_available()
                
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
            self.use_gpu_augmentation = bool(use_gpu_augmentation)
            
            if self.use_gpu_augmentation:
                if not is_kornia_available():
                    if auto_install_kornia:
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
        
        self.use_trivialaugment = use_trivialaugment
        self.use_randaugment = use_randaugment
        self.rand_n = rand_n
        self.rand_m = rand_m
        self.color_jitter = color_jitter
        self.random_erase_p = random_erase_p

    def build_transforms(self, train=True):
        """
        Build transform pipeline based on augmentation_mode.
        
        Augmentation intensity (CPU or GPU):
        - OFF: No augmentation (ToTensor, Normalize only)
        - MINIMAL: RandomCrop, RandomHorizontalFlip
        - BASIC: + ColorJitter (50% prob)
        - STRONG: + Rotation, RandAugment/TrivialAugment, RandomErasing
        
        Note: If use_gpu_augmentation=True, CPU transforms are minimal and
              heavy augmentations are applied on GPU via GPUAugmentation.
        """
        ops = []
        
        if train:
            if self.augmentation_mode == AugmentationMode.OFF:
                # No augmentation - just resize and crop
                ops.append(transforms.Resize(int(self.image_size * 1.14)))
                ops.append(transforms.CenterCrop(self.image_size))
                
            elif self.augmentation_mode == AugmentationMode.MINIMAL:
                # MINIMAL: Only fast augmentations
                ops.append(transforms.Resize(int(self.image_size * 1.14)))
                ops.append(transforms.RandomCrop(self.image_size))
                ops.append(transforms.RandomHorizontalFlip(p=0.5))
                
            elif self.augmentation_mode == AugmentationMode.BASIC:
                # BASIC: Standard augmentations for production
                ops.append(transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.0), ratio=(3./4., 4./3.)))
                ops.append(transforms.RandomHorizontalFlip(p=0.5))
                if self.color_jitter:
                    ops.append(transforms.RandomApply([transforms.ColorJitter(*self.color_jitter)], p=0.5))
                
            elif self.augmentation_mode == AugmentationMode.STRONG:
                # STRONG: Maximum augmentations for research/accuracy
                ops.append(transforms.RandomResizedCrop(self.image_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)))
                ops.append(transforms.RandomHorizontalFlip(p=0.5))
                if self.color_jitter:
                    ops.append(transforms.RandomApply([transforms.ColorJitter(*self.color_jitter)], p=0.8))
                ops.append(transforms.RandomRotation(degrees=10))
                
                # Add advanced augmentations
                if self.use_trivialaugment:
                    try:
                        ops.append(transforms.TrivialAugmentWide())
                    except AttributeError:
                        ops.append(transforms.RandAugment(num_ops=self.rand_n, magnitude=self.rand_m))
                elif self.use_randaugment:
                    ops.append(transforms.RandAugment(num_ops=self.rand_n, magnitude=self.rand_m))
                
                # RandomErasing
                if self.random_erase_p > 0:
                    # Apply before ToTensor if needed, or after - we'll do after
                    pass  # Will be added after ToTensor below
        else:
            # Validation/test transforms (same for all modes)
            ops.append(transforms.Resize(int(self.image_size * 1.14)))
            ops.append(transforms.CenterCrop(self.image_size))
            
        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(self.mean, self.std))
        
        # RandomErasing only in STRONG mode (applied after ToTensor)
        if train and self.random_erase_p > 0 and self.augmentation_mode == AugmentationMode.STRONG:
            ops.append(transforms.RandomErasing(p=self.random_erase_p, scale=(0.02, 0.33), ratio=(0.3, 3.3)))
            
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
                device=device
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
            transform=self.build_transforms(train=True)
        )
        test_dataset = self.dataset_cls(
            root=self.data_dir,
            train=False,
            download=self.download,
            transform=self.build_transforms(train=False)
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
