# LayerZero

A modular PyTorch training framework with automatic performance optimizations.

## Features

### Trainer
- Model compilation via `torch.compile()` (PyTorch 2.0+)
- Mixed precision training (AMP)
- Automatic GPU augmentation integration
- Asynchronous CUDA data transfers
- **Real-time TensorBoard logging**
- **PyTorch Profiler integration** (GPU/CPU/memory analysis)
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

Monitoring & Analysis:
- Real-time TensorBoard logging (loss, metrics, lr)
- PyTorch Profiler integration (find bottlenecks)
- GPU/CPU utilization tracking
- Memory usage profiling

---

## Installation

```bash
pip install torch torchvision matplotlib tensorboard torch-tb-profiler

# Optional: GPU augmentation
pip install kornia kornia-rs
```

Or install from PyPI:

```bash
pip install LayerZero
```

**Note:** `torch-tb-profiler` is required to view PyTorch Profiler traces in TensorBoard.

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
    download=True,
    use_gpu_augmentation='auto'  # Automatic GPU acceleration
)

train_loader, test_loader = loader.get_loaders()

# Training configuration
config = TrainerConfig(
    epochs=10,
    amp=True,
    compile_model='auto',
    use_tensorboard=True  # TensorBoard enabled by default!
)

# Train
trainer = Trainer(
    model=model,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    config=config
)

results = trainer.fit(
    train_loader, 
    test_loader,
    data_loader=loader  # Auto-detects GPU augmentation!
)
```

**View Training in Real-Time:**

```python
# Google Colab / Kaggle (inline in notebook - two separate commands):
%load_ext tensorboard
%tensorboard --logdir runs
```

```bash
# Local / Terminal:
tensorboard --logdir=runs
# Then open: http://localhost:6006
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
# Automatic integration with Trainer (Recommended)
loader = ImageDataLoader(
    CIFAR10,
    use_gpu_augmentation='auto',  # Auto-detect GPU and Kornia
    auto_install_kornia=True       # Install if missing
)

train_loader, test_loader = loader.get_loaders()

# GPU augmentation auto-detected when fit() is called!
trainer = Trainer(
    model=model,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    config=config
)

trainer.fit(
    train_loader,
    test_loader,
    data_loader=loader  # ← Pass loader here, Trainer auto-detects GPU aug!
)

# Manual usage in custom training loops
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

### TensorBoard (Real-Time Monitoring) 📊

**Automatically enabled by default!** Works seamlessly in Google Colab, Kaggle, and local environments.

#### 🎯 Google Colab / Kaggle Usage (Recommended)

```python
# Step 1: Load TensorBoard extension (run once at top of notebook)
%load_ext tensorboard

# Step 2: Train your model (TensorBoard logs automatically)
trainer = Trainer(model, loss_fn, optimizer, config=TrainerConfig(epochs=10))
trainer.fit(train_loader, val_loader)

# Step 3: View TensorBoard inline in your notebook
%tensorboard --logdir runs
```

**That's it!** TensorBoard will display directly in your Colab/Kaggle notebook with real-time updates.

#### 💻 Local / Terminal Usage

```bash
# Terminal 1: Start training
python train.py

# Terminal 2: Start TensorBoard
tensorboard --logdir=runs

# Open browser to: http://localhost:6006
```

#### ⚙️ Configuration

```python
config = TrainerConfig(
    use_tensorboard=True,              # Enable/disable (default: True)
    tensorboard_log_dir="runs",        # Log directory
    tensorboard_comment="experiment1", # Experiment name/tag
    tensorboard_log_graph=True,        # Log model graph
    tensorboard_log_gradients=False,   # Log gradient histograms (slower)
)
```

#### 📈 What Gets Logged

- ✅ Train & validation losses (real-time, per epoch)
- ✅ All custom metrics (accuracy, F1, etc.)
- ✅ Learning rate changes over time
- ✅ Model graph visualization (optional)
- ✅ Gradient & weight histograms (optional)
- ✅ **PyTorch Profiler** (optional - GPU/CPU utilization, memory, bottlenecks)

#### 🔧 Advanced Options

**Disable TensorBoard:**

```python
config = TrainerConfig(
    use_tensorboard=False  # Turn off TensorBoard logging
)
```

**Multiple experiments with names:**

```python
# Experiment 1
config1 = TrainerConfig(tensorboard_comment="resnet50_lr0.001")
trainer1.fit(train_loader, val_loader)

# Experiment 2
config2 = TrainerConfig(tensorboard_comment="resnet50_lr0.01")
trainer2.fit(train_loader, val_loader)

# View both: %tensorboard --logdir runs
```

**Manual callback control:**

```python
from LayerZero import Trainer, TensorBoardCallback

tb_callback = TensorBoardCallback(
    log_dir="my_experiments",
    comment="custom_experiment",
    log_gradients=True  # Enable gradient logging
)

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    config=TrainerConfig(use_tensorboard=False),  # Disable auto-init
    callbacks=[tb_callback]  # Add manually
)
```

#### 📱 Colab/Kaggle Quick Start

```python
# Complete Colab/Kaggle example
%load_ext tensorboard

from LayerZero import ImageDataLoader, Trainer, TrainerConfig
from torchvision.datasets import CIFAR10
import torch.nn as nn

# Setup model and data
model = nn.Sequential(...)
loader = ImageDataLoader(CIFAR10, root='./data', batch_size=128)
train_loader, val_loader = loader.get_loaders()

# Train with TensorBoard (automatic)
trainer = Trainer(
    model=model,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    config=TrainerConfig(epochs=10)  # TensorBoard enabled by default!
)

trainer.fit(train_loader, val_loader, data_loader=loader)

# View results inline
%tensorboard --logdir runs
```

#### 🔬 PyTorch Profiler Integration (Performance Analysis)

**NEW!** Analyze GPU/CPU utilization, memory usage, and identify bottlenecks - all in TensorBoard!

```python
# Enable profiler with TensorBoard
config = TrainerConfig(
    epochs=10,
    use_tensorboard=True,
    use_profiler=True,  # Enable PyTorch Profiler
)

trainer = Trainer(model, loss_fn, optimizer, config=config)
trainer.fit(train_loader, val_loader)

# View profiler traces in TensorBoard
%tensorboard --logdir runs
# Look for the "PYTORCH_PROFILER" or "PROFILE" tab (requires torch-tb-profiler)
```

**What you'll see:**
- 📊 GPU/CPU utilization timeline
- 💾 Memory usage over time (allocated/reserved)
- ⚡ Operation timing breakdown
- 🔍 Bottleneck identification (slow ops highlighted)
- 📈 Kernel execution trace

**Requirements:**
- `torch-tb-profiler` must be installed: `pip install torch-tb-profiler`
- The profiler tab will appear after profiling data is generated

**Fine-tune profiler schedule:**

```python
config = TrainerConfig(
    use_profiler=True,
    profiler_schedule_wait=1,      # Skip first N batches
    profiler_schedule_warmup=1,    # Warmup for N batches
    profiler_schedule_active=3,    # Profile for N batches
    profiler_schedule_repeat=2,    # Repeat cycle N times
)
```

**Why use the profiler?**
- Find GPU idle time (data loading bottlenecks)
- Identify slow operations
- Optimize memory usage
- Compare different model architectures
- Debug performance issues

**⚠️ Performance Note:**
- TensorBoard (default): < 1% overhead ✅
- Gradient logging: ~5-10% overhead (disabled by default)
- Profiler: ~10-15% overhead (disabled by default)
- Logging happens once per epoch, not per batch
- Safe to keep TensorBoard enabled for all training

**Example: Optimizing based on profiler insights**

```python
# Before profiling: Found data loading is slow
# Solution: Increase num_workers

loader = ImageDataLoader(
    CIFAR10,
    batch_size=128,
    num_workers=4,  # Increased from default
    use_gpu_augmentation='auto'  # Move augmentation to GPU
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
    amp=True,                          # Mixed precision
    compile_model='auto',              # torch.compile()
    compile_mode='default',
    device='auto',
    save_dir='./checkpoints',
    # TensorBoard settings
    use_tensorboard=True,              # Enable TensorBoard (default: True)
    tensorboard_log_dir='runs',        # TensorBoard log directory
    tensorboard_comment='',            # Experiment name/comment
    tensorboard_log_graph=True,        # Log model graph
    tensorboard_log_gradients=False,   # Log gradient histograms
    # PyTorch Profiler settings (integrates with TensorBoard)
    use_profiler=False,                # Enable PyTorch Profiler (default: False)
    profiler_schedule_wait=1,          # Batches to skip before profiling
    profiler_schedule_warmup=1,        # Warmup batches
    profiler_schedule_active=3,        # Active profiling batches
    profiler_schedule_repeat=2,        # Number of profiling cycles
)
```

### Trainer

```python
Trainer(
    model,
    loss_fn,
    optimizer,
    config,
    metrics=None,
    callbacks=None,
)

# Run training with optional GPU augmentation auto-detection
trainer.fit(
    train_loader, 
    val_loader,
    epochs=None,        # Optional: Override config.epochs
    data_loader=None    # Optional: ImageDataLoader for GPU aug auto-detection
)

trainer.evaluate(dataloader)  # Evaluate on data
trainer.predict(dataloader)   # Get predictions
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
