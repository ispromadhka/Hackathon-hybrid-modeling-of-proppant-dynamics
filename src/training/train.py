"""
Training script for SuperB-FNO proppant model.
Supports resume/fine-tuning from existing checkpoint.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import argparse
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.fno import create_model
from src.training.dataset import ProppantDataset, create_dataloaders


class CombinedLoss(nn.Module):
    """Combined loss: MSE + Relative L2 + Temporal consistency."""

    def __init__(self, mse_weight: float = 1.0, rel_weight: float = 0.5, temporal_weight: float = 0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.rel_weight = rel_weight
        self.temporal_weight = temporal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = torch.mean((pred - target) ** 2)

        diff_norm = torch.norm(pred - target, p=2)
        target_norm = torch.norm(target, p=2) + 1e-6
        rel_loss = diff_norm / target_norm

        if pred.shape[1] > 1:
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            temporal_loss = torch.mean((pred_diff - target_diff) ** 2)
        else:
            temporal_loss = torch.tensor(0.0, device=pred.device)

        return self.mse_weight * mse_loss + self.rel_weight * rel_loss + self.temporal_weight * temporal_loss


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute evaluation metrics."""
    with torch.no_grad():
        mae = torch.mean(torch.abs(pred - target)).item()
        rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
        rel_l2 = (torch.norm(pred - target) / (torch.norm(target) + 1e-6)).item()
        max_err = torch.max(torch.abs(pred - target)).item()
        accuracy = (torch.abs(pred - target) < 0.05).float().mean().item() * 100
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - target.mean()) ** 2) + 1e-6
        r2 = (1 - ss_res / ss_tot).item()

    return {
        'mae': mae, 'rmse': rmse, 'rel_l2': rel_l2,
        'max_err': max_err, 'accuracy': accuracy, 'r2': max(0, r2 * 100),
    }


class Trainer:
    """Training manager with resume support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        lr: float = 1e-3,
        n_epochs: int = 100,
        checkpoint_dir: Path = None,
        patience: int = 8,
        min_delta: float = 1e-4,
        resume_path: str = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_epochs = n_epochs
        self.start_epoch = 1

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        warmup_epochs = min(5, max(1, n_epochs // 10))
        self.warmup_epochs = warmup_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, n_epochs - warmup_epochs), eta_min=lr * 0.01
        ) if n_epochs > warmup_epochs else None

        self.criterion = CombinedLoss(mse_weight=1.0, rel_weight=0.5, temporal_weight=0.1)

        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_accuracy = 0.0
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0

        # Resume from checkpoint
        if resume_path:
            self._load_checkpoint(resume_path)

    def _load_checkpoint(self, path: str):
        """Load checkpoint for fine-tuning."""
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            print(f"[RESUME] Checkpoint not found: {path}")
            return

        print(f"[RESUME] Loading from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"[RESUME] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[RESUME] Unexpected keys: {len(unexpected)}")

        # Optionally load optimizer
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("[RESUME] Optimizer state loaded")
            except:
                print("[RESUME] Could not load optimizer, using fresh")

        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']

        print("[RESUME] Model loaded successfully")

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(params)

            n_times = pred.shape[1]
            target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)

            loss = self.criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        all_metrics = []

        for batch in self.val_loader:
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)

            pred = self.model(params)
            n_times = pred.shape[1]
            target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)

            loss = self.criterion(pred, target)
            total_loss += loss.item()
            all_metrics.append(compute_metrics(pred, target))

        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
        return avg_loss, avg_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False, metrics: dict = None):
        if not self.checkpoint_dir:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
        }

        torch.save(checkpoint, self.checkpoint_dir / 'last.pt')
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')

    def train(self):
        print(f"Training on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 100)
        print(f"{'Epoch':>5} | {'Train':>10} | {'Val':>10} | {'MAE':>8} | {'Acc%':>6} | {'R²%':>6} | {'LR':>9} | {'Time':>6} | Status")
        print("-" * 100)

        initial_lr = self.optimizer.param_groups[0]['lr']

        for epoch in range(self.start_epoch, self.n_epochs + 1):
            t0 = time.time()

            if epoch <= self.warmup_epochs:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = initial_lr * epoch / self.warmup_epochs

            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()

            if epoch > self.warmup_epochs and self.scheduler:
                self.scheduler.step()

            is_best = val_loss < (self.best_val_loss - self.min_delta)
            if is_best:
                self.best_val_loss = val_loss
                self.best_accuracy = metrics['accuracy']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best, metrics)

            status = "[BEST]" if is_best else f"[{self.patience_counter}/{self.patience}]" if self.patience_counter > 0 else ""
            print(f"{epoch:5d} | {train_loss:10.4e} | {val_loss:10.4e} | {metrics['mae']:8.4f} | "
                  f"{metrics['accuracy']:6.1f} | {metrics['r2']:6.1f} | {self.optimizer.param_groups[0]['lr']:9.2e} | "
                  f"{time.time()-t0:5.1f}s | {status}")

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping after {epoch} epochs")
                break

        print("-" * 100)
        print(f"Best validation loss: {self.best_val_loss:.4e}, Best accuracy: {self.best_accuracy:.1f}%")


def main(epochs: int = 100, lr: float = 1e-3, patience: int = 8, resume: str = None):
    """Main training entry point."""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not (data_dir / 'metadata.json').exists():
        print(f"No data found in {data_dir}")
        return

    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    nx, ny = metadata['nx'], metadata['ny']
    n_times_meta = metadata.get('n_times', 201)

    batch_size = 8
    if nx * ny * n_times_meta >= 2_000_000:
        batch_size = 4
    if nx * ny * n_times_meta >= 4_000_000:
        batch_size = 2

    train_loader, val_loader = create_dataloaders(data_dir, batch_size=batch_size, train_ratio=0.8, num_workers=0)

    sample = next(iter(train_loader))
    n_times = sample['trajectory'].shape[1]
    n_params = sample['params'].shape[1]

    mode = "RESUME" if resume else "NEW"
    print(f"[{mode}] Training: {epochs} epochs, lr={lr}, patience={patience}")
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Grid: {nx}x{ny}, {n_times} steps, {n_params} params")

    model = create_model(nx=nx, ny=ny, n_times=n_times, n_params=n_params, device=device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        n_epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        resume_path=resume,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SuperB-FNO')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for fine-tuning')
    args = parser.parse_args()

    main(epochs=args.epochs, lr=args.lr, patience=args.patience, resume=args.resume)
