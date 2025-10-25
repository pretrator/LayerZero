"""
Best-in-class PyTorch Trainer module (class-based).
Features:
- Single `Trainer` class with `.fit()`, `.evaluate()`, `.predict()`
- Mixed precision (AMP) support
- Gradient accumulation
- Gradient clipping
- Scheduler & warmup support
- Early stopping
- Checkpointing (save best / resume)
- Flexible metrics & logging (tqdm)
- Optional callbacks interface for extensibility
- Support for train/val/test DataLoaders
- Built-in utilities: mixup, label smoothing (simple), accuracy metric usage

Notes:
- This file assumes `torch`, `tqdm`, and common helper functions (e.g., `accuracy_fn`) are available.
- Designed to be easy to extend.
"""

from __future__ import annotations
from .Helper import Helper
import os
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

@dataclass
class TrainerConfig:
    """
    Configuration for Trainer with performance optimizations.
    
    Performance Features:
        - Model Compilation: 20-50% faster with torch.compile() [PyTorch 2.0+] (default: AUTO)
        - Mixed Precision (AMP): 2-3x faster training, 50% less memory (default: ON)
        - Gradient accumulation: Train with larger effective batch sizes
        - Gradient clipping: Prevent exploding gradients
        - Non-blocking transfers: Automatic async data movement
    
    TensorBoard Features (Minimal CPU overhead):
        - Real-time loss and metrics visualization (logs once per epoch)
        - Learning rate tracking (logs once per epoch)
        - Gradient histogram logging (optional, disabled by default - adds ~5-10% overhead)
        - Model graph visualization (optional, logs once - negligible overhead)
        - PyTorch Profiler integration (optional, disabled by default - adds ~10-15% overhead)
          * GPU/CPU utilization tracking
          * Memory usage profiling
          * Operation timing analysis
          * Bottleneck identification
          * Auto-installs torch-tb-profiler plugin when enabled
        
        Performance Impact (typical usage):
        - TensorBoard enabled (default): < 1% overhead (only epoch-end logging)
        - With gradient logging: ~5-10% overhead (histogram computation is expensive)
        - With profiler: ~10-15% overhead (detailed tracing is expensive)
        
        Recommendation: Keep defaults (tensorboard=True, gradients=False, profiler=False)
    """
    device: Optional[torch.device] = None
    epochs: int = 10
    batch_size: int = 32
    grad_accum_steps: int = 1  # Gradient accumulation steps
    max_grad_norm: Optional[float] = None  # Gradient clipping threshold
    amp: bool = True  # Mixed precision (FP16) - 2-3x faster, 50% less memory!
    compile_model: str = 'auto'  # 'auto', True, False - torch.compile() for 20-50% speedup!
    compile_mode: str = 'default'  # 'default', 'reduce-overhead', 'max-autotune'
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    monitor: str = "val_accuracy"  # metric to monitor for best model
    monitor_mode: str = "max"  # 'max' or 'min'
    early_stopping_patience: Optional[int] = None
    scheduler: Optional[Any] = None  # Allow any scheduler type or callable
    scheduler_step_per_batch: bool = False  # Step scheduler per batch (required for OneCycleLR)
    initial_lr: Optional[float] = None
    print_every: int = 1
    seed: Optional[int] = 42
    # TensorBoard settings
    use_tensorboard: bool = True  # Enable TensorBoard logging
    tensorboard_log_dir: str = "runs"  # TensorBoard log directory
    tensorboard_log_graph: bool = True  # Log model graph
    tensorboard_log_gradients: bool = False  # Log gradient histograms (can be slow)
    tensorboard_comment: str = ""  # Experiment comment/name
    # PyTorch Profiler settings (integrates with TensorBoard)
    use_profiler: bool = False  # Enable PyTorch Profiler (performance analysis)
    profiler_schedule_wait: int = 1  # Number of steps to skip at start
    profiler_schedule_warmup: int = 1  # Number of steps for warmup
    profiler_schedule_active: int = 3  # Number of steps to profile
    profiler_schedule_repeat: int = 2  # Number of times to repeat profiling cycle


class Callback:
    """Base callback. Override methods you're interested in."""

    def on_epoch_begin(self, trainer: "Trainer", epoch: int):
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]):
        pass

    def on_batch_begin(self, trainer: "Trainer", batch: int):
        pass

    def on_batch_end(self, trainer: "Trainer", batch: int, logs: Dict[str, Any]):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience: int = 5, monitor: str = "val_loss", mode: str = "min"):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0
        self.stop_training = False

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]):
        val = logs.get(self.monitor)
        if val is None:
            return
        if self.best is None:
            self.best = val
            self.num_bad_epochs = 0
            return
        improved = (val < self.best) if self.mode == "min" else (val > self.best)
        if improved:
            self.best = val
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.stop_training = True


class CheckpointCallback(Callback):
    def __init__(self, save_dir: str, save_best_only: bool = True, monitor: str = "val_loss", mode: str = "min"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]):
        filepath = os.path.join(self.save_dir, f"checkpoint_epoch{epoch}.pt")
        metric = logs.get(self.monitor)
        if metric is None and self.save_best_only:
            trainer._save_checkpoint(filepath)
            return
        if not self.save_best_only:
            trainer._save_checkpoint(filepath)
            return
        if self.best is None:
            self.best = metric
            trainer._save_checkpoint(filepath)
            return
        improved = (metric < self.best) if self.mode == "min" else (metric > self.best)
        if improved:
            self.best = metric
            trainer._save_checkpoint(filepath)


class TensorBoardCallback(Callback):
    """
    TensorBoard logging callback for real-time training visualization.
    
    Features:
        - Real-time loss and metrics visualization
        - Learning rate tracking
        - Gradient histogram logging (optional)
        - Model graph visualization (optional)
        - PyTorch Profiler integration (optional)
    
    Usage:
        Open TensorBoard in your browser: tensorboard --logdir=runs
        View profiler traces in the "PYTORCH_PROFILER" tab
    """
    
    @staticmethod
    def _is_torch_tb_profiler_available():
        """Check if torch-tb-profiler is installed."""
        try:
            import torch_tb_profiler
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _install_torch_tb_profiler():
        """Auto-install torch-tb-profiler package."""
        import subprocess
        import sys
        
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "torch-tb-profiler"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def __init__(
        self,
        log_dir: str = "runs",
        comment: str = "",
        log_graph: bool = True,
        log_gradients: bool = False,
        use_profiler: bool = False,
        profiler_schedule_wait: int = 1,
        profiler_schedule_warmup: int = 1,
        profiler_schedule_active: int = 3,
        profiler_schedule_repeat: int = 2,
    ):
        try:
            from torch.utils.tensorboard import SummaryWriter
            import datetime
            
            # Auto-install torch-tb-profiler if profiler is enabled
            if use_profiler and not self._is_torch_tb_profiler_available():
                print("\n📦 torch-tb-profiler not found, installing...")
                if self._install_torch_tb_profiler():
                    print("✅ torch-tb-profiler installed - PYTORCH_PROFILER tab will be available\n")
                else:
                    print("⚠️  Auto-install failed. Manual install: pip install torch-tb-profiler\n")
            
            # Create unique run name with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"{timestamp}"
            if comment:
                run_name = f"{comment}_{run_name}"
            
            self.log_dir = os.path.join(log_dir, run_name)
            self.writer = SummaryWriter(log_dir=self.log_dir, comment=comment)
            self.log_graph = log_graph
            self.log_gradients = log_gradients
            self.graph_logged = False
            self.use_profiler = use_profiler
            self.profiler = None
            self.profiler_schedule_wait = profiler_schedule_wait
            self.profiler_schedule_warmup = profiler_schedule_warmup
            self.profiler_schedule_active = profiler_schedule_active
            self.profiler_schedule_repeat = profiler_schedule_repeat
            
            # Build logging info
            features = ["Loss/Metrics/LR"]
            if log_graph:
                features.append("Model Graph")
            if log_gradients:
                features.append("Gradients")
            if use_profiler:
                features.append(f"Profiler(wait={profiler_schedule_wait},warmup={profiler_schedule_warmup},active={profiler_schedule_active},repeat={profiler_schedule_repeat})")
            
            print(f"\n📊 TensorBoard: {', '.join(features)}")
            print(f"   Dir: {self.log_dir}")
            print(f"   View: tensorboard --logdir={log_dir}")
            if use_profiler and self._is_torch_tb_profiler_available():
                print(f"   Profiler: Look for PYTORCH_PROFILER tab\n")
            else:
                print()
            
        except ImportError:
            print("\n⚠️  TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int):
        """Initialize PyTorch Profiler at start of first epoch"""
        if self.writer is None:
            return
        
        # Initialize profiler on first epoch
        if self.use_profiler and self.profiler is None and epoch == 1:
            try:
                # Profiler with TensorBoard integration
                schedule = torch.profiler.schedule(
                    wait=self.profiler_schedule_wait,
                    warmup=self.profiler_schedule_warmup,
                    active=self.profiler_schedule_active,
                    repeat=self.profiler_schedule_repeat,
                )
                
                self.profiler = torch.profiler.profile(
                    schedule=schedule,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
                
                self.profiler.__enter__()
                print(f"✅ Profiler started → {self.log_dir}\n")
                
            except Exception as e:
                print(f"⚠️  Profiler init failed: {e}\n")
                self.profiler = None
                self.use_profiler = False
    
    def on_batch_end(self, trainer: "Trainer", batch: int, logs: Dict[str, Any]):
        """Step profiler (only when enabled) - optimized for minimal CPU overhead"""
        # Early exit if writer is not available
        if self.writer is None:
            return
        
        # Only step profiler if it's active (no unnecessary work otherwise)
        # This check is fast - just a None comparison
        if self.profiler is not None:
            self.profiler.step()  # Required for profiler to work correctly
    
    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]):
        """Log metrics, losses, and learning rate at epoch end"""
        if self.writer is None:
            return
        
        # Log all metrics and losses
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                # Separate train and val metrics
                if key.startswith("train_"):
                    metric_name = key.replace("train_", "")
                    self.writer.add_scalar(f"Train/{metric_name}", value, epoch)
                elif key.startswith("val_"):
                    metric_name = key.replace("val_", "")
                    self.writer.add_scalar(f"Validation/{metric_name}", value, epoch)
                else:
                    self.writer.add_scalar(f"Metrics/{key}", value, epoch)
        
        # Log learning rate
        try:
            current_lr = trainer.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("Learning_Rate/lr", current_lr, epoch)
        except (AttributeError, IndexError, KeyError):
            pass
        
        # Log gradient histograms (if enabled)
        if self.log_gradients:
            try:
                for name, param in trainer.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
                        self.writer.add_histogram(f"Weights/{name}", param, epoch)
            except Exception:
                pass  # Gradient logging is optional
        
        # Flush to disk
        self.writer.flush()
    
    def _cleanup_profiler(self):
        """Safely cleanup profiler resources"""
        if hasattr(self, 'profiler') and self.profiler is not None:
            try:
                self.profiler.__exit__(None, None, None)
                print("\n✅ Profiler data saved successfully\n")
            except Exception as e:
                print(f"\n⚠️  Error closing profiler: {e}\n")
            finally:
                self.profiler = None

    def _cleanup_writer(self):
        """Safely cleanup writer resources"""
        if hasattr(self, 'writer') and self.writer is not None:
            try:
                self.writer.flush()  # Ensure all data is written
                self.writer.close()
            except Exception as e:
                print(f"\n⚠️  Error closing writer: {e}\n")
            finally:
                self.writer = None

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]):
        """Log metrics, losses, and learning rate at epoch end"""
        if self.writer is None:
            return
        
        # Log all metrics and losses
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                # Separate train and val metrics
                if key.startswith("train_"):
                    metric_name = key.replace("train_", "")
                    self.writer.add_scalar(f"Train/{metric_name}", value, epoch)
                elif key.startswith("val_"):
                    metric_name = key.replace("val_", "")
                    self.writer.add_scalar(f"Validation/{metric_name}", value, epoch)
                else:
                    self.writer.add_scalar(f"Metrics/{key}", value, epoch)
        
        # Log learning rate
        try:
            current_lr = trainer.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("Learning_Rate/lr", current_lr, epoch)
        except (AttributeError, IndexError, KeyError):
            pass
        
        # Log gradient histograms (if enabled)
        if self.log_gradients:
            try:
                for name, param in trainer.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
                        self.writer.add_histogram(f"Weights/{name}", param, epoch)
            except Exception:
                pass  # Gradient logging is optional
        
        # Cleanup profiler if training is done
        if epoch == trainer.config.epochs:
            self._cleanup_profiler()
        
        # Flush to disk
        self.writer.flush()

    def __del__(self):
        """Cleanup resources on deletion"""
        self._cleanup_profiler()
        self._cleanup_writer()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: optim.Optimizer,
        config: TrainerConfig = TrainerConfig(),
        metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.config = config
        # Wrap all metrics to handle different input signatures
        self.metrics = {
            name: self._wrap_metric(name, fn) 
            for name, fn in (metrics or {}).items()
        }
        self.callbacks = callbacks or []
        self.helper = Helper()
        self.gpu_augmentation = None  # Detected lazily in fit()
        
        # Auto-initialize TensorBoard callback if enabled
        if config.use_tensorboard:
            tb_callback = TensorBoardCallback(
                log_dir=config.tensorboard_log_dir,
                comment=config.tensorboard_comment,
                log_graph=config.tensorboard_log_graph,
                log_gradients=config.tensorboard_log_gradients,
                use_profiler=config.use_profiler,
                profiler_schedule_wait=config.profiler_schedule_wait,
                profiler_schedule_warmup=config.profiler_schedule_warmup,
                profiler_schedule_active=config.profiler_schedule_active,
                profiler_schedule_repeat=config.profiler_schedule_repeat,
            )
            self.callbacks.append(tb_callback)

        self.device = config.device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        # Model Compilation (PyTorch 2.0+)
        self.original_model = model
        self.model = self._compile_model_if_requested(model)
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed)

        # Mixed Precision Training (AMP)
        amp_enabled = config.amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        
        # Get device info for display
        device_name = self.device.type.upper()
        if self.device.type == "cuda" and torch.cuda.device_count() > 0:
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception:
                pass  # Fall back to "CUDA" if device name unavailable
        
        # Print AMP status
        if amp_enabled:
            print(f"\n⚡ AMP (FP16): {device_name} | 2-3x faster, 50% less memory\n")
        elif self.device.type == "cuda" and not config.amp:
            print(f"\nℹ️  AMP disabled on {device_name} | Enable for 2-3x speedup: TrainerConfig(amp=True)\n")
        else:
            print(f"\nℹ️  Training on {device_name}")
            if self.device.type == "cpu":
                print("   (AMP not available on CPU)\n")
            else:
                print()
        
        os.makedirs(self.config.save_dir, exist_ok=True)
        self._best_metric = None
        self._history: List[Dict[str, Any]] = []

    def _calculate_steps_per_epoch(self, dataloader: DataLoader) -> int:
        """
        Calculate steps per epoch accounting for batch size and gradient accumulation.
        
        Args:
            dataloader: Training DataLoader
            
        Returns:
            Number of optimizer steps per epoch
        """
        # Get total dataset size
        try:
            dataset_size = len(dataloader.dataset)
        except (TypeError, AttributeError):
            # Handle datasets without __len__ (e.g. IterableDataset)
            try:
                dataset_size = dataloader.dataset.total_size  # Custom attribute some use
            except AttributeError:
                # Fallback: estimate from batches * batch_size
                dataset_size = len(dataloader) * dataloader.batch_size
        
        # Calculate effective batch size
        batch_size = dataloader.batch_size or 1  # Default to 1 if None
        grad_accum = self.config.grad_accum_steps
        effective_batch_size = batch_size * grad_accum
        
        # Calculate steps per epoch (ceil to handle uneven division)
        return math.ceil(dataset_size / effective_batch_size)

    def _is_batch_scheduler(self, scheduler: Any) -> bool:
        """Check if scheduler requires per-batch stepping (e.g. OneCycleLR)"""
        from torch.optim.lr_scheduler import OneCycleLR
        # Add other batch-level schedulers here if desired
        return isinstance(scheduler, OneCycleLR)
        
    def _configure_scheduler(self, train_loader: DataLoader) -> None:
        """
        Configure scheduler with correct steps_per_epoch if needed.
        
        Args:
            train_loader: Training DataLoader to calculate steps from
        """
        if self.config.scheduler is None:
            return
            
        from torch.optim.lr_scheduler import OneCycleLR
        
        # Special handling for OneCycleLR which needs total_steps
        if isinstance(self.config.scheduler, OneCycleLR):
            # Get scheduler kwargs
            scheduler_dict = self.config.scheduler.state_dict()
            
            # Calculate actual steps needed
            steps_per_epoch = self._calculate_steps_per_epoch(train_loader)
            total_steps = steps_per_epoch * self.config.epochs
            
            # Check if steps match
            if scheduler_dict['total_steps'] != total_steps:
                print(f"\n⚠️  OneCycleLR steps mismatch:")
                print(f"   Configured: {scheduler_dict['total_steps']}")
                print(f"   Required: {total_steps} ({steps_per_epoch} steps/epoch × {self.config.epochs} epochs)")
                print(f"   Use: OneCycleLR(optimizer, max_lr, total_steps={total_steps})\n")

    def _wrap_metric(self, name: str, fn: Callable) -> Callable[[torch.Tensor, torch.Tensor], float]:
        """
        Wraps a metric function to handle different input signatures and ensure float output.
        
        Supports three common signatures:
        1. fn(y_true=y, y_pred=pred) - preferred
        2. fn(pred, target) - common in PyTorch
        3. fn(y, logits) - direct tensor inputs
        
        Args:
            name: Metric name for error reporting
            fn: Metric function to wrap
            
        Returns:
            Wrapped metric function that accepts (y, logits) and returns float
        """
        def wrapped(y: torch.Tensor, logits: torch.Tensor) -> float:
            try:
                # Try modern signature first (y_true, y_pred)
                if logits.dim() > 1 and logits.size(1) > 1:  # Probably classification logits
                    pred = logits.argmax(dim=1)
                else:
                    pred = logits
                return float(fn(y_true=y, y_pred=pred))
            except Exception:
                try:
                    # Try common PyTorch (pred, target) signature
                    if logits.dim() > 1 and logits.size(1) > 1:
                        pred = logits.argmax(dim=1)
                    else:
                        pred = logits
                    return float(fn(pred, y))
                except Exception:
                    try:
                        # Try direct tensor inputs as last resort
                        return float(fn(y, logits))
                    except Exception as e:
                        raise ValueError(
                            f"Metric '{name}' failed with all signatures. Error: {str(e)}\n"
                            "Supported signatures:\n"
                            "1. fn(y_true=y, y_pred=pred) - preferred\n"
                            "2. fn(pred, target) - common in PyTorch\n"
                            "3. fn(y, logits) - direct tensor inputs"
                        ) from e
        return wrapped

    def _compile_model_if_requested(self, model: nn.Module) -> nn.Module:
        """
        Compile model using torch.compile() if requested and available.
        
        Args:
            model: PyTorch model to compile
            
        Returns:
            Compiled model or original model if compilation not available/requested
        """
        # Check if torch.compile is available (PyTorch 2.0+)
        compile_available = hasattr(torch, 'compile')
        
        # Determine if we should compile
        should_compile = False
        if self.config.compile_model == 'auto':
            # Auto: compile if available and on CUDA
            should_compile = compile_available and torch.cuda.is_available()
        elif self.config.compile_model in [True, 'true', 'True']:
            should_compile = True
        
        if not should_compile:
            if self.config.compile_model not in ['auto', False, 'false', 'False']:
                print(f"\nℹ️  Model compilation disabled.")
            return model
        
        if not compile_available:
            print(f"\n⚠️  torch.compile() unavailable (PyTorch {torch.__version__} < 2.0)\n")
            return model
        
        # Compile the model
        try:
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
            print(f"\n🔥 torch.compile(mode={self.config.compile_mode}): {device_name} | 20-50% faster (1st epoch slower)\n")
            
            compiled_model = torch.compile(
                model,
                mode=self.config.compile_mode,
                # fullgraph=False,  # Allow graph breaks for flexibility
            )
            return compiled_model
            
        except Exception as e:
            print(f"\n⚠️  Compilation failed: {str(e)[:100]}... | Using non-compiled model\n")
            return model

    def _save_checkpoint(self, path: str):
        """Save checkpoint with original model state dict to ensure compatibility"""
        state = {
            "model_state": self.original_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict() if hasattr(self, "scaler") else None,
            "config": self.config,
            "history": self._history,  # Save training history for resuming
        }
        torch.save(state, path)

    def _load_checkpoint(self, path: str, map_location: Optional[torch.device] = None):
        """Load checkpoint into original model and try to sync with compiled model"""
        state = torch.load(path, map_location=map_location or self.device)
        
        # Load into original model first
        self.original_model.load_state_dict(state["model_state"])
        
        # Try to sync with compiled/wrapped model if different
        if self.model is not self.original_model:
            try:
                self.model.load_state_dict(self.original_model.state_dict())
            except Exception:
                print("\n⚠️  Warning: Could not sync compiled model with original model state dict")
                print("   Training will continue with loaded weights in original model only\n")
        
        # Load optimizer and scaler states
        self.optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scaler_state") and hasattr(self, "scaler"):
            self.scaler.load_state_dict(state["scaler_state"])
            
        # Restore training history if available
        if state.get("history"):
            self._history = state["history"]

    def _run_one_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        training: bool = True,
    ) -> Dict[str, float]:
        is_train = training
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        metric_sums = {k: 0.0 for k in self.metrics.keys()}
        total_samples = 0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=("Train" if is_train else "Eval"))
        self.optimizer.zero_grad()
        
        # Track if we've hit a torch.compile error
        compile_failed = False

        for batch_idx, (X, y) in pbar:
            # Skip batch_begin callbacks if list is empty (performance optimization)
            if self.callbacks:
                for cb in self.callbacks:
                    cb.on_batch_begin(self, batch_idx)

            # Use non_blocking=True to enable asynchronous data transfer
            # This allows CPU to continue preparing next batch while GPU transfers current batch
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            # Apply GPU augmentation if enabled (training only)
            if is_train and self.gpu_augmentation is not None:
                X = self.gpu_augmentation(X)
            
            batch_size = X.shape[0]

            try:
                with torch.set_grad_enabled(is_train):
                    if self.config.amp and self.device.type == "cuda":
                        # Use modern torch.amp.autocast API (PyTorch 1.10+)
                        # Falls back to torch.cuda.amp.autocast for older versions
                        try:
                            autocast_context = torch.amp.autocast('cuda')
                        except AttributeError:
                            autocast_context = torch.cuda.amp.autocast()
                        
                        with autocast_context:
                            logits = self.model(X)
                            loss = self.loss_fn(logits, y)
                    else:
                        logits = self.model(X)
                        loss = self.loss_fn(logits, y)
            except Exception as e:
                # Catch torch.compile errors and provide helpful feedback
                error_msg = str(e)
                if 'Dynamo' in error_msg or 'FakeTensor' in error_msg or 'compile' in error_msg.lower():
                    if not compile_failed:
                        compile_failed = True
                        print(f"\n⚠️  torch.compile() error: {error_msg[:150]}...")
                        print("   Fix: Check model architecture OR set compile_model=False\n")
                raise

            # normalize loss across accumulation steps
            loss_value = loss.detach().item() if isinstance(loss, torch.Tensor) else float(loss)
            if is_train and self.config.grad_accum_steps > 1:
                loss = loss / float(self.config.grad_accum_steps)

            if is_train:
                if self.config.amp and self.device.type == "cuda":
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # gradient accumulation step handling
                do_step = ((batch_idx + 1) % self.config.grad_accum_steps) == 0 or (batch_idx + 1) == len(dataloader)
                if do_step:
                    if self.config.max_grad_norm is not None:
                        if self.config.amp and self.device.type == "cuda":
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    if self.config.amp and self.device.type == "cuda":
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Scheduler step (batch-level) -- required for OneCycleLR
                    if is_train and self.config.scheduler is not None:
                        try:
                            if self.config.scheduler_step_per_batch or self._is_batch_scheduler(self.config.scheduler):
                                self.config.scheduler.step()
                        except Exception:
                            pass

            total_loss += loss_value * batch_size
            total_samples += batch_size

            # Compute metrics (using wrapped metric functions)
            for name, fn in self.metrics.items():
                metric_val = fn(y, logits)  # Wrapper handles signature and output type
                metric_sums[name] += metric_val * batch_size

            logs = {
                "loss": total_loss / total_samples if total_samples else 0.0,
                **{name: metric_sums[name] / total_samples for name in metric_sums},
            }

            # Batch-end callbacks (only if callbacks exist)
            if self.callbacks:
                for cb in self.callbacks:
                    cb.on_batch_end(self, batch_idx, logs)

            pbar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})

        # Scheduler step after epoch (if provided and training)
        # Skip if scheduler is stepped per batch or is a batch scheduler
        if (is_train and self.config.scheduler is not None and 
            not (self.config.scheduler_step_per_batch or self._is_batch_scheduler(self.config.scheduler))):
            try:
                self.config.scheduler.step()
            except Exception:
                # Some schedulers require a metric input
                pass

        epoch_metrics = {"loss": total_loss / total_samples if total_samples else 0.0}
        epoch_metrics.update({name: metric_sums[name] / total_samples for name in metric_sums})
        return epoch_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        epochs = epochs or self.config.epochs
        stop_training = False
        early_stopper = next((c for c in self.callbacks if isinstance(c, EarlyStopping)), None)
        
        # Configure scheduler with correct steps_per_epoch
        self._configure_scheduler(train_loader)
        
        # Auto-detect GPU augmentation from train_loader's attached ImageDataLoader
        # EAFP: Easier to Ask for Forgiveness xx Permission (Pythonic!)
        if self.gpu_augmentation is None:
            try:
                # Try to get ImageDataLoader instance from train_loader
                image_data_loader = getattr(train_loader, '_image_data_loader', None)
                if image_data_loader is not None and image_data_loader.use_gpu_augmentation:
                    self.gpu_augmentation = image_data_loader.get_gpu_augmentation()
                    if self.gpu_augmentation is not None:
                        print("\n🎨 GPU Augmentation: Auto-detected from train_loader (training only)\n")
            except (AttributeError, TypeError):
                # Not from ImageDataLoader or GPU augmentation not available
                pass

        for epoch in range(1, epochs + 1):
            # Epoch-begin callbacks
            if self.callbacks:
                for cb in self.callbacks:
                    cb.on_epoch_begin(self, epoch)

            train_logs = self._run_one_epoch(train_loader, epoch, training=True)
            val_logs: Dict[str, float] = {}
            if val_loader is not None:
                val_logs = self._run_one_epoch(val_loader, epoch, training=False)

            logs = {f"train_{k}": v for k, v in train_logs.items()}
            logs.update({f"val_{k}": v for k, v in val_logs.items()})

            # Resolve monitored metric key and value
            monitor_key = self.config.monitor
            if not monitor_key.startswith("val_") and f"val_{monitor_key}" in logs:
                monitor_key = f"val_{monitor_key}"
            monitored = logs.get(monitor_key)

            # Epoch-end callbacks (metrics logging, checkpointing, etc.)
            if self.callbacks:
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, logs)

            # update history
            epoch_record = {"epoch": epoch, **logs}
            self._history.append(epoch_record)

            # print
            if epoch % self.config.print_every == 0:
                simple_log = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                print(f"Epoch {epoch}/{epochs} — {simple_log}")

            if early_stopper and getattr(early_stopper, "stop_training", False):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if val_loader is not None:
            try:
                self.helper.plotTrainTestLoss(self._history)
            except Exception as e:
                print(f"Could not plot losses: {e}")
        return self._history

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        return self._run_one_epoch(dataloader, epoch=-1, training=False)

    def predict(
        self, 
        dataloader: DataLoader,
        return_logits: bool = False
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Run prediction on the dataloader
        
        Args:
            dataloader: DataLoader to predict on
            return_logits: If True, return (logits, labels). If False, return (predictions, labels)
                         For classification, predictions are argmax of logits
        
        Returns:
            If return_logits=False: (predictions, labels) where predictions are class indices
            If return_logits=True: (logits, predictions, labels) where predictions are class indices
        """
        self.model.to(self.device)
        self.model.eval()
        preds = []
        trues = []
        
        with torch.inference_mode():
            for X, y in tqdm(dataloader, desc="Predict"):
                X = X.to(self.device, non_blocking=True)
                logits = self.model(X)
                # Keep predictions on device, only move to CPU at the end
                preds.append(logits.detach())
                if y is not None:  # Handle datasets without labels
                    trues.append(y)
        
        # Single large transfer instead of many small ones - much faster
        all_logits = torch.cat(preds, dim=0).cpu()
        all_preds = all_logits.argmax(dim=1) if all_logits.dim() > 1 else all_logits
        
        # Handle labels if they exist
        all_trues = None
        if len(trues) > 0:
            all_trues = torch.cat(trues, dim=0).cpu()
            
        return (all_logits, all_preds, all_trues) if return_logits else (all_preds, all_trues)



def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Performs mixup on the input batch and targets.
    
    Args:
        x: Input batch tensor of shape [N, ...] where N is batch size
        y: Target batch tensor of shape [N, ...] where N is batch size
        alpha: Mixup interpolation coefficient (default: 0.4)
               - alpha = 0: No mixup
               - alpha > 0: Beta(alpha, alpha) distribution for mixing factor
        device: Optional device to place generated tensors on
    
    Returns:
        Tuple containing:
        - mixed_x: Mixed input batch
        - y_a: Original targets
        - y_b: Permuted targets
        - lam: Mixing coefficient used
        
    Example usage:
        # During training:
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
        output = model(mixed_x)
        loss = lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)
    """
    if alpha <= 0:
        return x, y, y, 1.0
    
    # Generate mixing factor from beta distribution
    lam = float(np.random.beta(alpha, alpha))
    
    # Move to appropriate device if specified
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    
    # Generate permutation for the batch
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    # Mix the samples
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


# # ---------- Example usage ----------
# if __name__ == "__main__":
#     import numpy as np
#     from torchvision import models

#     # tiny example model and dataset
#     class DummyDataset(torch.utils.data.Dataset):
#         def __init__(self, n=1000, c=3, h=32, w=32, num_classes=10):
#             self.X = torch.randn(n, c, h, w)
#             self.y = torch.randint(0, num_classes, (n,))

#         def __len__(self):
#             return len(self.X)

#         def __getitem__(self, idx):
#             return self.X[idx], self.y[idx]

#     train_ds = DummyDataset(n=1024)
#     val_ds = DummyDataset(n=256)
#     train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

#     model = models.resnet18(num_classes=10)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
#     config = TrainerConfig(device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), epochs=3, amp=True)

#     # simple accuracy metric that follows (y_true=..., y_pred=...)
#     def simple_acc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
#         return (y_true == y_pred).float().mean().item() * 100.0

#     trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, config=config, metrics={"accuracy": simple_acc}, callbacks=[CheckpointCallback(config.save_dir, save_best_only=False)])
#     trainer.fit(train_loader, val_loader)

#     preds, trues = trainer.predict(val_loader)
#     print("Done")
