"""
Enhanced training script for FNO v2 proppant model.

Improvements:
1. Large batch size (32-64) with gradient accumulation
2. Mixed precision training (AMP) for speed
3. Advanced loss function (L2 + Spectral + H1 + Temporal)
4. Learning rate warmup + cosine annealing
5. Gradient clipping
6. EMA (Exponential Moving Average) for stable inference
7. Multiple metrics tracking
8. SpecBoost training (optional)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import json
import time
import copy
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.fno_v2 import (
    create_enhanced_model,
    CombinedLossV2,
    EnhancedFNOProppant,
    SpecBoostFNO
)
from src.training.dataset import ProppantDataset, create_dataloaders
from src.training.train import Metrics  # Reuse metrics from train.py


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class EnhancedTrainer:
    """
    Enhanced training manager for FNO v2.

    Features:
    - Large batch training with gradient accumulation
    - Mixed precision (AMP)
    - EMA weights
    - Advanced loss function
    - Comprehensive metrics
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        lr: float = 1e-3,
        n_epochs: int = 500,  # More epochs for large models
        checkpoint_dir: Path = None,
        patience: int = 50,   # More patience
        min_delta: float = 1e-5,
        use_amp: bool = True,  # Mixed precision
        gradient_accumulation_steps: int = 4,  # Effective batch size = batch_size * 4
        ema_decay: float = 0.999,
        weight_decay: float = 0.01,
        warmup_epochs: int = 10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_epochs = n_epochs
        self.use_amp = use_amp and device == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_epochs = warmup_epochs

        # Optimizer: AdamW with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler: warmup + cosine annealing
        total_steps = n_epochs * len(train_loader)
        warmup_steps = warmup_epochs * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Loss function
        self.criterion = CombinedLossV2(
            l2_weight=0.4,
            spectral_weight=0.3,
            h1_weight=0.2,
            temporal_weight=0.1,
        )

        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # EMA
        self.ema = EMA(model, decay=ema_decay)

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # History
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        self.best_val_loss = float('inf')
        self.best_metrics = {}

        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.early_stop = False

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(pbar):
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    pred = self.model(params)
                    n_times = pred.shape[1]
                    target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)
                    loss = self.criterion(pred, target)
                    loss = loss / self.gradient_accumulation_steps

                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                pred = self.model(params)
                n_times = pred.shape[1]
                target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)
                loss = self.criterion(pred, target)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

                # Update EMA
                self.ema.update()

            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation_steps:.4e}'})

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate using EMA weights."""
        self.model.eval()

        # Apply EMA weights for validation
        self.ema.apply_shadow()

        total_loss = 0.0
        batch_metrics = []

        for batch in self.val_loader:
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)

            if self.use_amp:
                with autocast():
                    pred = self.model(params)
                    n_times = pred.shape[1]
                    target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)
                    loss = self.criterion(pred, target)
            else:
                pred = self.model(params)
                n_times = pred.shape[1]
                target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)
                loss = self.criterion(pred, target)

            total_loss += loss.item()

            # Compute metrics
            metrics = Metrics.compute_all(pred.float(), target.float())
            batch_metrics.append(metrics)

        # Restore original weights
        self.ema.restore()

        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = Metrics.aggregate(batch_metrics)

        return avg_loss, avg_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if not self.checkpoint_dir:
            return

        # Save with EMA weights
        self.ema.apply_shadow()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        torch.save(checkpoint, self.checkpoint_dir / 'last_v2.pt')
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_v2.pt')

        self.ema.restore()

    def train(self):
        print("=" * 100)
        print("Enhanced FNO v2 Training")
        print("=" * 100)
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps}x")
        print(f"Effective Batch Size: {self.train_loader.batch_size * self.gradient_accumulation_steps}")
        print(f"Early Stopping: patience={self.patience}, min_delta={self.min_delta}")
        print(f"Warmup Epochs: {self.warmup_epochs}")
        print("-" * 100)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.metrics_history.append(val_metrics)

            # Check improvement
            is_best = val_loss < (self.best_val_loss - self.min_delta)
            if is_best:
                self.best_val_loss = val_loss
                self.best_metrics = val_metrics
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            status = ""
            if is_best:
                status = " [BEST]"
            elif self.patience_counter > 0:
                status = f" [{self.patience_counter}/{self.patience}]"

            # Print progress
            print(
                f"Epoch {epoch:4d}/{self.n_epochs} | "
                f"Train: {train_loss:.4e} | "
                f"Val: {val_loss:.4e} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
                + status
            )
            print(f"         → {Metrics.format(val_metrics)}")

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                self.early_stop = True
                break

        print("=" * 100)
        print("Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4e}")
        print(f"Best Metrics: {Metrics.format(self.best_metrics)}")
        print("=" * 100)


import math


def main(
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 32,   # Larger batch size
    model_size: str = 'large',
    use_specboost: bool = False,
    patience: int = 50,
):
    """Main training entry point for FNO v2."""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Check if data exists
    if not (data_dir / 'metadata.json').exists():
        print(f"No data found in {data_dir}")
        print("Run: python app.py --generate --samples 500")
        return

    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    # Create dataloaders with larger batch size
    train_loader, val_loader = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        train_ratio=0.8,
        num_workers=4 if device == 'cuda' else 0
    )

    # Get dimensions from data
    sample = next(iter(train_loader))
    n_times = sample['trajectory'].shape[1]
    n_params = sample['params'].shape[1]

    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Grid: {metadata['nx']}x{metadata['ny']}, {n_times} time steps, {n_params} params")
    print(f"Model size: {model_size}, SpecBoost: {use_specboost}")

    # Create model
    model = create_enhanced_model(
        nx=metadata['nx'],
        ny=metadata['ny'],
        n_times=n_times,
        n_params=n_params,
        device=device,
        model_size=model_size,
        use_specboost=use_specboost,
    )

    # Create trainer
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        n_epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        use_amp=device == 'cuda',
        gradient_accumulation_steps=4,  # Effective batch = 32 * 4 = 128
    )

    trainer.train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train FNO v2')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model-size', choices=['small', 'medium', 'large', 'xlarge'], default='large')
    parser.add_argument('--specboost', action='store_true')
    parser.add_argument('--patience', type=int, default=50)

    args = parser.parse_args()

    main(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        model_size=args.model_size,
        use_specboost=args.specboost,
        patience=args.patience,
    )
