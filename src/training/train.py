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


import torch.nn.functional as F

class RelativeLpLoss(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        return torch.mean(diff_norms / (y_norms + 1e-4))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        rel_loss = self.rel(pred, target)
        mse_loss = F.mse_loss(pred, target)

        neg_penalty = torch.mean(F.relu(-pred)**2) * 10.0

        n_cells = pred.shape[2] * pred.shape[3]
        mass_pred = torch.sum(pred, dim=(2, 3)) / n_cells
        mass_target = torch.sum(target, dim=(2, 3)) / n_cells
        mass_loss = F.mse_loss(mass_pred, mass_target)

        return rel_loss + 5.0 * mse_loss + neg_penalty + mass_loss


class Trainer:
    """Training manager for FNO model with early stopping."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        lr: float = 1e-3,
        n_epochs: int = 100,
        checkpoint_dir: Path = None,
        patience: int = 15,  # Early stopping patience
        min_delta: float = 1e-4  # Minimum improvement threshold
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_epochs = n_epochs

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        warmup_epochs = min(5, max(1, n_epochs // 10))
        self.warmup_epochs = warmup_epochs
        self.scheduler = None
        if n_epochs > warmup_epochs:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=n_epochs - warmup_epochs,
                eta_min=lr * 0.01
            )

        # Relative L2 loss (better for PDEs than MSE)
        self.criterion = RelativeLpLoss(p=2)

        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.early_stop = False

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
            target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)

            loss = self.criterion(pred, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)

            pred = self.model(params)

            n_times = pred.shape[1]
            target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)

            total_loss += self.criterion(pred, target).item()

        return total_loss / len(self.val_loader)

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

    def train(self, n_epochs: int = None):
        if n_epochs is None:
            n_epochs = self.n_epochs

        print(f"Training on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Scheduler: {self.warmup_epochs} warmup epochs + cosine annealing")
        print(f"Early stopping: patience={self.patience}, min_delta={self.min_delta}")
        print("-" * 80)

        initial_lr = self.optimizer.param_groups[0]['lr']

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            # Warmup: linearly increase LR
            if epoch <= self.warmup_epochs:
                warmup_factor = epoch / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = initial_lr * warmup_factor

            train_loss = self.train_epoch()
            val_loss = self.validate()
            if epoch > self.warmup_epochs:
                if self.scheduler is not None:
                    self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Check for improvement
            is_best = val_loss < (self.best_val_loss - self.min_delta)
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            quality = max(0, (1 - val_loss) * 100)

            status = ""
            if is_best:
                status = " [BEST]"
            elif self.patience_counter > 0:
                status = f" [{self.patience_counter}/{self.patience}]"

            print(
                f"Epoch {epoch:3d} | "
                f"Train: {train_loss:.4e} | "
                f"Val: {val_loss:.4e} | "
                f"Quality: {quality:.1f}% | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
                + status
            )

            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {self.patience} epochs)")
                self.early_stop = True
                break

        print("-" * 80)
        print(f"Training finished!")
        print(f"Best validation loss: {self.best_val_loss:.4e}")

        self.model.eval()
        q_sum = 0
        with torch.no_grad():
            for batch in self.val_loader:
                p, t = batch['params'].to(self.device), batch['trajectory'].to(self.device)
                pred = self.model(p)
                target = t[:, :pred.shape[1], :, :].permute(0, 1, 3, 2)
                mae = torch.mean(torch.abs(pred - target))
                q_sum += max(0, (1 - mae.item()) * 100)
        print(f"Best quality: {q_sum/len(self.val_loader):.1f}%")


def main(epochs: int = 100, lr: float = 1e-3, patience: int = 15):
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

    batch_size = 8
    try:
        nx = int(metadata.get('nx', 0))
        ny = int(metadata.get('ny', 0))
        n_times_meta = int(metadata.get('n_times', 0))
        # Reduce batch size if memory is an issue
        if nx * ny * max(n_times_meta, 1) >= 2_000_000:
            batch_size = 2
        if nx * ny * max(n_times_meta, 1) >= 4_000_000:
            batch_size = 1
    except Exception:
        batch_size = 1

    train_loader, val_loader = create_dataloaders(
        data_dir, batch_size=batch_size, train_ratio=0.8, num_workers=0
    )

    # Get n_times and n_params from data
    sample = next(iter(train_loader))
    n_times = sample['trajectory'].shape[1]
    n_params = sample['params'].shape[1]

    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Grid: {metadata['nx']}x{metadata['ny']}, {n_times} time steps, {n_params} params")

    model = create_model(
        nx=metadata['nx'],
        ny=metadata['ny'],
        n_times=n_times,
        n_params=n_params,
        device=device
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        n_epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        patience=patience
    )

    trainer.train()


if __name__ == '__main__':
    main()
