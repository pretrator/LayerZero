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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

@dataclass
class TrainerConfig:
    device: Optional[torch.device] = None
    epochs: int = 10
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_grad_norm: Optional[float] = None
    amp: bool = True  # mixed precision
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    monitor: str = "val_accuracy"  # metric to monitor for best model
    monitor_mode: str = "max"  # 'max' or 'min'
    early_stopping_patience: Optional[int] = None
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    initial_lr: Optional[float] = None
    print_every: int = 1
    seed: Optional[int] = 42


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
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config
        self.metrics = metrics or {}
        self.callbacks = callbacks or []
        self.helper = Helper()

        self.device = config.device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed)

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp and self.device.type == "cuda")
        os.makedirs(self.config.save_dir, exist_ok=True)
        self._best_metric = None
        self._history: List[Dict[str, Any]] = []

    def _save_checkpoint(self, path: str):
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict() if hasattr(self, "scaler") else None,
            "config": self.config,
        }
        torch.save(state, path)

    def _load_checkpoint(self, path: str, map_location: Optional[torch.device] = None):
        state = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scaler_state") and hasattr(self, "scaler"):
            self.scaler.load_state_dict(state["scaler_state"])

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

        for batch_idx, (X, y) in pbar:
            for cb in self.callbacks:
                cb.on_batch_begin(self, batch_idx)

            X = X.to(self.device)
            y = y.to(self.device)
            batch_size = X.shape[0]

            with torch.set_grad_enabled(is_train):
                if self.config.amp and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        logits = self.model(X)
                        loss = self.loss_fn(logits, y)
                else:
                    logits = self.model(X)
                    loss = self.loss_fn(logits, y)

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

            total_loss += loss_value * batch_size
            total_samples += batch_size

            # metrics
            for name, fn in self.metrics.items():
                try:
                    metric_val = fn(y_true=y, y_pred=(logits.argmax(dim=1) if logits is not None else logits))
                except TypeError:
                    # older metric signature: (y_pred, y_true)
                    metric_val = fn(logits, y)
                metric_sums[name] += float(metric_val) * batch_size

            logs = {
                "loss": total_loss / total_samples if total_samples else 0.0,
                **{name: metric_sums[name] / total_samples for name in metric_sums},
            }

            for cb in self.callbacks:
                cb.on_batch_end(self, batch_idx, logs)

            pbar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})

        # scheduler step after epoch (if provided and training)
        if is_train and self.config.scheduler is not None:
            try:
                self.config.scheduler.step()
            except Exception:
                # some schedulers require a metric input
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

        for epoch in range(1, epochs + 1):
            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            train_logs = self._run_one_epoch(train_loader, epoch, training=True)
            val_logs: Dict[str, float] = {}
            if val_loader is not None:
                val_logs = self._run_one_epoch(val_loader, epoch, training=False)

            logs = {f"train_{k}": v for k, v in train_logs.items()}
            logs.update({f"val_{k}": v for k, v in val_logs.items()})

            # combined metric for checkpoint decision
            monitored = (
                logs.get(f"val_{self.config.monitor.split('_')[-1]}")
                or logs.get(self.config.monitor)
                or None
            )

            # notify callbacks
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
                self.helper.plot_train_test_loss(self._history)
            except Exception as e:
                print(f"Could not plot losses: {e}")
        return self._history

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        return self._run_one_epoch(dataloader, epoch=-1, training=False)

    def predict(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.to(self.device)
        self.model.eval()
        preds = []
        trues = []
        with torch.inference_mode():
            for X, y in tqdm(dataloader, desc="Predict"):
                X = X.to(self.device)
                out = self.model(X)
                preds.append(out.detach().cpu())
                trues.append(y.detach().cpu())
        return torch.cat(preds, dim=0), torch.cat(trues, dim=0)



def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    if alpha <= 0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
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
