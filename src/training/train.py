"""
Training script for FNO proppant model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.fno import create_model
from src.training.dataset import ProppantDataset, create_dataloaders


class RelativeLpLoss(nn.Module):
    """Relative Lp loss for PDE learning."""

    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        diff_norm = torch.norm(pred_flat - target_flat, p=self.p, dim=1)
        target_norm = torch.norm(target_flat, p=self.p, dim=1) + 1e-8

        return (diff_norm / target_norm).mean()


class Trainer:
    """Training manager for FNO model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        lr: float = 1e-3,
        checkpoint_dir: Path = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)

        self.criterion = RelativeLpLoss(p=2)
        self.mse = nn.MSELoss()

        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(params)

            # Match dimensions
            n_times = pred.shape[1]
            target = trajectory[:, :n_times, :, :]

            loss = self.criterion(pred, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0

        for batch in self.val_loader:
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)

            pred = self.model(params)

            n_times = pred.shape[1]
            target = trajectory[:, :n_times, :, :]

            total_loss += self.criterion(pred, target).item()
            total_mse += self.mse(pred, target).item()

        n = len(self.val_loader)
        return {'rel_l2': total_loss / n, 'mse': total_mse / n}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if not self.checkpoint_dir:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }

        torch.save(checkpoint, self.checkpoint_dir / 'last.pt')
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')

    def train(self, n_epochs: int):
        print(f"Training on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch()
            val_metrics = self.validate()

            self.scheduler.step()

            is_best = val_metrics['rel_l2'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['rel_l2']

            self.save_checkpoint(epoch, is_best)

            elapsed = time.time() - t0
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

        print(f"\nBest val loss: {self.best_val_loss:.4e}")


def main(epochs: int = 100):
    """Main training entry point."""
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

    train_loader, val_loader = create_dataloaders(
        data_dir, batch_size=16, train_ratio=0.8, num_workers=0
    )

    # Get n_times from data
    sample = next(iter(train_loader))
    n_times = sample['trajectory'].shape[1]

    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Grid: {metadata['nx']}x{metadata['ny']}, {n_times} time steps")

    model = create_model(
        nx=metadata['nx'],
        ny=metadata['ny'],
        n_times=n_times,
        device=device
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        checkpoint_dir=checkpoint_dir
    )

    trainer.train(n_epochs=epochs)


if __name__ == '__main__':
    main()
