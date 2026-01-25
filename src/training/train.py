"""
Training script for FNO model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Optional, Dict
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.fno import FNOTrajectoryModel, create_model
from src.training.dataset import ProppantDataset, create_dataloaders


class RelativeLpLoss(nn.Module):
    """Relative Lp loss for PDE learning."""

    def __init__(self, p: int = 2, reduction: str = 'mean'):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dimensions
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        # Compute relative error
        diff_norm = torch.norm(pred_flat - target_flat, p=self.p, dim=1)
        target_norm = torch.norm(target_flat, p=self.p, dim=1)

        rel_error = diff_norm / (target_norm + 1e-8)

        if self.reduction == 'mean':
            return rel_error.mean()
        elif self.reduction == 'sum':
            return rel_error.sum()
        return rel_error


class Trainer:
    """Training manager for FNO model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )

        self.criterion = RelativeLpLoss(p=2)
        self.mse_criterion = nn.MSELoss()

        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            c0 = batch['c0'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)
            u = batch['u'].to(self.device)
            v = batch['v'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(c0, u, v)

            # Loss on full trajectory (skip initial condition)
            loss = self.criterion(pred, trajectory[..., 1:, :, :].transpose(1, 2).transpose(2, 3))

            # Actually, let's match dimensions properly
            # trajectory: (batch, n_times, nx, ny)
            # pred: (batch, n_times, nx, ny)
            # We want to predict from t=1 onwards, so compare pred with trajectory[1:]
            if pred.shape[1] == trajectory.shape[0] - 1:
                target = trajectory[:, 1:, :, :]
            else:
                target = trajectory

            loss = self.criterion(pred, target)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0

        for batch in self.val_loader:
            c0 = batch['c0'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)
            u = batch['u'].to(self.device)
            v = batch['v'].to(self.device)

            pred = self.model(c0, u, v)

            if pred.shape[1] == trajectory.shape[1] - 1:
                target = trajectory[:, 1:, :, :]
            else:
                target = trajectory

            loss = self.criterion(pred, target)
            mse = self.mse_criterion(pred, target)

            total_loss += loss.item()
            total_mse += mse.item()

        n = len(self.val_loader)
        return {
            'rel_l2': total_loss / n,
            'mse': total_mse / n
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        torch.save(checkpoint, self.checkpoint_dir / 'last.pt')

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['epoch']

    def train(self, n_epochs: int):
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, n_epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_metrics = self.validate()

            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['rel_l2'])

            is_best = val_metrics['rel_l2'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['rel_l2']

            self.save_checkpoint(epoch, is_best)

            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch:3d} | "
                f"Train: {train_loss:.4e} | "
                f"Val: {val_metrics['rel_l2']:.4e} | "
                f"MSE: {val_metrics['mse']:.4e} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
                + (" *" if is_best else "")
            )

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4e}")


def main():
    """Main training entry point."""
    # Config
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load metadata
    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir,
        batch_size=8,
        train_ratio=0.8,
        num_workers=0  # Set to 0 for debugging
    )

    # Determine n_times from data
    sample = next(iter(train_loader))
    n_times = sample['trajectory'].shape[1]

    print(f"Data loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Grid: {metadata['nx']}x{metadata['ny']}, {n_times} time steps")

    # Create model
    model = create_model(
        nx=metadata['nx'],
        ny=metadata['ny'],
        n_times=n_times,
        device=device
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        checkpoint_dir=checkpoint_dir
    )

    trainer.train(n_epochs=100)


if __name__ == '__main__':
    main()
