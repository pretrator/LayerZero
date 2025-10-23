# LayerZero

A modular PyTorch training framework with automatic performance optimizations.

## Features

### Trainer
- Model compilation via `torch.compile()` (PyTorch 2.0+)
- Mixed precision training (AMP)
- Asynchronous CUDA data transfers
- Metric tracking and logging
- Model checkpointing
- Custom callbacks

### ImageDataLoader
- GPU-accelerated augmentation using Kornia
- Configurable augmentation modes
- Automatic worker detection
- Torchvision dataset support
- Pinned memory for GPU training

### Helper
- Training/validation metric tracking
- Loss curve visualization
- Experiment logging

---

## Performance Optimizations

Applied automatically:
- `torch.compile()` for model compilation
- Mixed precision (FP16) training
- Non-blocking CUDA transfers
- GPU-based augmentation (Kornia)
- Optimized DataLoader configuration

---

## Installation

```bash
pip install torch torchvision matplotlib

# Optional: GPU augmentation
pip install kornia kornia-rs
```

---

## Usage

### Basic Example

```python
import torch
from torch import nn
from LayerZero import ImageDataLoader, Trainer, TrainerConfig
from torchvision.datasets import CIFAR10

# Model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3*32*32, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Data
loader = ImageDataLoader(
    CIFAR10,
    root='./data',
    image_size=32,
    batch_size=128,
    download=True
)

train_loader, test_loader = loader.get_dataloaders()

# Training configuration
config = TrainerConfig(
    epochs=10,
    amp=True,
    compile_model='auto'
)

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    config=config,
)

results = trainer.train()
```

---

## Configuration

### Augmentation Modes

```python
from LayerZero import ImageDataLoader, AugmentationMode

loader = ImageDataLoader(
    CIFAR10,
    augmentation_mode=AugmentationMode.MINIMAL,  # Flip + Crop
    # AugmentationMode.BASIC,   # + ColorJitter (default)
    # AugmentationMode.STRONG,  # + Rotation + Blur + Erasing
    # AugmentationMode.OFF,     # No augmentation
)
```

### GPU Augmentation

```python
loader = ImageDataLoader(
    CIFAR10,
    use_gpu_augmentation='auto',  # Auto-detect
    auto_install_kornia=True       # Install if missing
)

# Manual usage
gpu_aug = loader.get_gpu_augmentation(device='cuda')

for X, y in train_loader:
    X = X.to(device)
    X = gpu_aug(X)
    # ... training code ...
```

### Mixed Precision

```python
config = TrainerConfig(
    amp=True,   # Enable (default)
    # amp=False  # Disable for debugging
)
```

### Model Compilation

```python
config = TrainerConfig(
    compile_model='auto',        # Auto-detect PyTorch 2.0+
    compile_mode='default',      # Compilation mode
    # compile_mode='reduce-overhead'
    # compile_mode='max-autotune'
)
```

### Custom Metrics

```python
def accuracy_fn(y_pred, y_true):
    return (y_pred.argmax(1) == y_true).float().mean().item() * 100

config = TrainerConfig(
    metrics={'accuracy': accuracy_fn}
)
```

### Callbacks

```python
def save_checkpoint(model, epoch, metrics):
    torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')

config = TrainerConfig(
    callbacks={'on_epoch_end': save_checkpoint}
)
```

---

## API Reference

### ImageDataLoader

```python
ImageDataLoader(
    dataset_cls,                          # Torchvision dataset class
    root='./data',                        # Data directory
    image_size=224,                       # Image size
    batch_size=64,                        # Batch size
    augmentation_mode=AugmentationMode.BASIC,
    use_gpu_augmentation='auto',
    auto_install_kornia=True,
    num_workers=None,                     # Auto-detect
    download=False,
)
```

### TrainerConfig

```python
TrainerConfig(
    epochs=10,
    amp=True,                     # Mixed precision
    compile_model='auto',         # torch.compile()
    compile_mode='default',
    metrics={},
    callbacks={},
    device='auto',
    log_interval=100,
    save_dir='./checkpoints',
)
```

### Trainer

```python
Trainer(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    config,
)

trainer.train()                    # Run training
trainer.predict(dataloader)        # Get predictions
trainer.save_checkpoint(path)      # Save model
```

### KorniaHelper

```python
from LayerZero import (
    is_kornia_available,
    install_kornia,
    ensure_kornia,
    get_kornia_version,
)

if ensure_kornia(auto_install=True):
    # Kornia available
    pass
```

---

## Architecture

```
LayerZero/
├── Trainer.py              # Training loop
├── ImageDataLoader.py      # Data loading
├── GPUAugmentation.py      # Kornia augmentation
├── AugmentationMode.py     # Augmentation enums
├── KorniaHelper.py         # Kornia management
└── Helper.py               # Metrics tracking
```

---

## Troubleshooting

### Kornia installation fails
```bash
pip install kornia kornia-rs
```

### torch.compile not available
Requires PyTorch 2.0+:
```bash
pip install --upgrade torch torchvision
```

### Out of memory
Reduce batch size or enable mixed precision:
```python
config = TrainerConfig(amp=True)
```

### Slow on CPU
Use minimal augmentation:
```python
loader = ImageDataLoader(
    ...,
    augmentation_mode=AugmentationMode.MINIMAL
)
```

---

## Releasing New Versions

```bash
# Bump version (bug fixes: 0.1.3 → 0.1.4)
make bump-patch

# Push to trigger PyPI release
make release
```

See [RELEASE_WORKFLOW.md](RELEASE_WORKFLOW.md) for complete guide.

---

## License

MIT
