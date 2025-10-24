# 📊 TensorBoard in Google Colab/Kaggle - Quick Fix Guide

## ❌ Common Error

```python
# WRONG - This will fail:
%load_ext tensorboard && %tensorboard --logdir=runs
```

**Error:**
```
ModuleNotFoundError: No module named 'tensorboard && %tensorboard --logdir=runs'
```

---

## ✅ Correct Usage

### Method 1: Separate Commands (Recommended)

```python
# Step 1: Load the extension (run once)
%load_ext tensorboard

# Step 2: Train your model
from LayerZero import Trainer, TrainerConfig
trainer = Trainer(model, loss_fn, optimizer, config=TrainerConfig(epochs=10))
trainer.fit(train_loader, val_loader)

# Step 3: View TensorBoard (run in a separate cell)
%tensorboard --logdir runs
```

### Method 2: Two Lines in Same Cell

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

---

## 🎯 Complete Colab/Kaggle Example

```python
# Cell 1: Install and Import
!pip install LayerZero
!pip install torch-tb-profiler  # Required for profiler visualization
%load_ext tensorboard

# Cell 2: Setup
from LayerZero import ImageDataLoader, Trainer, TrainerConfig
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch

# Cell 3: Model and Data
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3*32*32, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

loader = ImageDataLoader(
    CIFAR10,
    root='./data',
    image_size=32,
    batch_size=128,
    download=True
)
train_loader, test_loader = loader.get_loaders()

# Cell 4: Train (TensorBoard logs automatically!)
config = TrainerConfig(
    epochs=10,
    use_tensorboard=True  # This is default!
)

trainer = Trainer(
    model=model,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    config=config
)

trainer.fit(train_loader, test_loader, data_loader=loader)

# Cell 5: View TensorBoard (inline in notebook)
%tensorboard --logdir runs
```

---

## 🔧 Troubleshooting

### Issue 1: "No module named 'tensorboard'"
```python
# Install tensorboard
!pip install tensorboard
```

### Issue 2: "No dashboards are active"
```python
# Make sure you've trained first, then check log directory
!ls runs/
```

### Issue 3: TensorBoard not updating
```python
# Reload TensorBoard
%reload_ext tensorboard
%tensorboard --logdir runs
```

### Issue 4: Multiple experiments not showing
```python
# TensorBoard shows all subdirectories in runs/
# Each training run creates a timestamped folder
!ls -la runs/
```

### Issue 5: Profiler tab not showing
```python
# Install the required plugin
!pip install torch-tb-profiler

# Restart TensorBoard after installing
%reload_ext tensorboard
%tensorboard --logdir runs
```

---

## 📈 What You'll See in TensorBoard

1. **SCALARS Tab** (Main tab)
   - Train/loss
   - Validation/loss
   - Train/accuracy
   - Validation/accuracy
   - Learning_Rate/lr

2. **GRAPHS Tab** (if enabled)
   - Model architecture visualization

3. **PYTORCH_PROFILER or PROFILE Tab** (if profiler enabled)
   - ⚠️ **Requires:** `pip install torch-tb-profiler`
   - GPU/CPU utilization
   - Memory usage
   - Operation timing

---

## 🎮 Tips for Colab/Kaggle

### Tip 1: Run TensorBoard in a Separate Cell
```python
# Cell 1: Load extension and train
%load_ext tensorboard
trainer.fit(train_loader, val_loader)

# Cell 2: View (run this in parallel with training or after)
%tensorboard --logdir runs
```

### Tip 2: Keep TensorBoard Cell Running
- The TensorBoard cell will keep updating in real-time
- Don't stop the cell while training
- You can scroll through epochs as they complete

### Tip 3: Multiple Experiments
```python
# Experiment 1
config1 = TrainerConfig(tensorboard_comment="lr_0.001")
trainer1.fit(train_loader, val_loader)

# Experiment 2
config2 = TrainerConfig(tensorboard_comment="lr_0.01")
trainer2.fit(train_loader, val_loader)

# View both
%tensorboard --logdir runs  # Shows all experiments!
```

### Tip 4: Refresh TensorBoard
```python
# If TensorBoard gets stuck, reload it
%reload_ext tensorboard
%tensorboard --logdir runs
```

---

## ⚡ Performance Notes

- TensorBoard adds < 1% overhead (default settings)
- Logging happens once per epoch (not per batch)
- Safe to keep enabled for all training
- No manual management needed

---

## 🚀 Advanced: Enable Profiler

**Step 1: Install the profiler plugin**
```python
!pip install torch-tb-profiler
```

**Step 2: Enable profiler in config**
```python
# Enable profiler for performance analysis
config = TrainerConfig(
    use_profiler=True,  # Adds ~10-15% overhead
    profiler_schedule_active=3,  # Profile 3 batches
)

trainer = Trainer(model, loss_fn, optimizer, config=config)
trainer.fit(train_loader, val_loader)
```

**Step 3: View profiler in TensorBoard**
```python
%tensorboard --logdir runs
# Look for the "PYTORCH_PROFILER" or "PROFILE" tab
```

**Note:** Without `torch-tb-profiler`, the profiler tab won't appear in TensorBoard!

---

## 📝 Key Takeaways

✅ **DO:**
- Use separate commands: `%load_ext tensorboard` then `%tensorboard --logdir runs`
- Load extension once at the start
- Keep TensorBoard cell running during training

❌ **DON'T:**
- Combine with `&&` - that's bash syntax, not Python
- Forget to load the extension first
- Stop TensorBoard cell while training

---

## 🎉 You're All Set!

TensorBoard will now display your training progress in real-time, right in your notebook!

