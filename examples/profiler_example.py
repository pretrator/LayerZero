"""
Example: Using PyTorch Profiler with TensorBoard

This example demonstrates how to enable the PyTorch Profiler to analyze
training performance. The torch-tb-profiler plugin will be auto-installed
if not already present.

Features:
- Auto-installation of torch-tb-profiler
- GPU/CPU utilization tracking
- Memory usage profiling
- Operation timing analysis
- TensorBoard visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from LayerZero import Trainer, TrainerConfig

# Create dummy dataset
def create_dummy_dataset(n_samples=1000, n_features=20, n_classes=10):
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(X, y)
    return dataset

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    # Create datasets
    train_dataset = create_dummy_dataset(n_samples=1000)
    val_dataset = create_dummy_dataset(n_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = SimpleModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Configure trainer with profiler enabled
    config = TrainerConfig(
        epochs=3,
        use_tensorboard=True,
        tensorboard_log_dir="runs",
        tensorboard_comment="profiler_test",
        # Enable profiler - torch-tb-profiler will be auto-installed!
        use_profiler=True,
        profiler_schedule_wait=1,
        profiler_schedule_warmup=1,
        profiler_schedule_active=3,
        profiler_schedule_repeat=2,
    )
    
    # Create trainer and fit
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
    )
    
    print("\n" + "="*60)
    print("🚀 Starting training with profiler enabled...")
    print("="*60)
    print("The torch-tb-profiler plugin will be auto-installed if needed.")
    print("\nAfter training completes:")
    print("1. Run: tensorboard --logdir=runs")
    print("2. Open: http://localhost:6006")
    print("3. Look for 'PYTORCH_PROFILER' or 'PROFILE' tab")
    print("="*60 + "\n")
    
    # Train
    trainer.fit(train_loader, val_loader)
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    print("View profiler results:")
    print("  tensorboard --logdir=runs")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

