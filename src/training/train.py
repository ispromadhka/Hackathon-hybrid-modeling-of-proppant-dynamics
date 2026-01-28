"""
Training script for FNO proppant model.
"""

import math
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


# ============== METRICS ==============
class Metrics:
    """
    Multiple metrics to track model quality during training.

    Metrics tracked:
    - Relative L2 Error (main loss)
    - MAE (Mean Absolute Error)
    - MSE (Mean Squared Error)
    - Max Error (worst case)
    - SSIM-like (Structural Similarity)
    - Mass Conservation Error
    - Coverage Accuracy (% of area with concentration > threshold)
    """

    @staticmethod
    @torch.no_grad()
    def compute_all(pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Compute all metrics for a batch."""
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        # Relative L2 Error
        diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
        target_norm = torch.norm(target_flat, p=2, dim=1) + 1e-8
        rel_l2 = (diff_norm / target_norm).mean().item()

        # MAE
        mae = torch.abs(pred - target).mean().item()

        # MSE
        mse = ((pred - target) ** 2).mean().item()

        # RMSE
        rmse = mse ** 0.5

        # Max Error
        max_err = torch.abs(pred - target).max().item()

        # Peak Signal-to-Noise Ratio (higher is better)
        max_val = max(target.max().item(), 1e-8)
        psnr = 10 * math.log10(max_val ** 2 / (mse + 1e-8)) if mse > 0 else 100.0

        # Mass Conservation Error (total mass should be similar)
        pred_mass = pred.sum(dim=(-2, -1)).mean()  # Sum over spatial, mean over batch/time
        target_mass = target.sum(dim=(-2, -1)).mean()
        if target_mass > 0.01:  # Only compute if there's significant mass
            mass_err = (torch.abs(pred_mass - target_mass) / target_mass).item()
        else:
            mass_err = 0.0  # No mass to conserve

        # Coverage Accuracy (% agreement on cells with c > 0.1)
        threshold = 0.1
        pred_coverage = (pred > threshold).float()
        target_coverage = (target > threshold).float()
        coverage_acc = (pred_coverage == target_coverage).float().mean().item() * 100

        # R² Score (Coefficient of Determination)
        ss_res = ((pred - target) ** 2).sum().item()
        ss_tot = ((target - target.mean()) ** 2).sum().item() + 1e-8
        r2 = 1 - ss_res / ss_tot

        return {
            'rel_l2': rel_l2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'max_err': max_err,
            'psnr': psnr,
            'mass_err': mass_err,
            'coverage_acc': coverage_acc,
            'r2': r2,
        }

    @staticmethod
    def aggregate(metrics_list: list) -> dict:
        """Aggregate metrics from multiple batches."""
        if not metrics_list:
            return {}

        keys = metrics_list[0].keys()
        aggregated = {}
        for key in keys:
            values = [m[key] for m in metrics_list]
            aggregated[key] = sum(values) / len(values)

        return aggregated

    @staticmethod
    def format(metrics: dict) -> str:
        """Format metrics for printing."""
        parts = []

        # Quality (inverse of rel_l2)
        quality = max(0, (1 - metrics.get('rel_l2', 1)) * 100)
        parts.append(f"Quality: {quality:.1f}%")

        # Key metrics
        if 'mae' in metrics:
            parts.append(f"MAE: {metrics['mae']:.4f}")
        if 'rmse' in metrics:
            parts.append(f"RMSE: {metrics['rmse']:.4f}")
        if 'max_err' in metrics:
            parts.append(f"MaxErr: {metrics['max_err']:.3f}")
        if 'r2' in metrics:
            parts.append(f"R²: {metrics['r2']:.3f}")
        if 'mass_err' in metrics:
            parts.append(f"MassErr: {metrics['mass_err']:.1%}")
        if 'coverage_acc' in metrics:
            parts.append(f"Coverage: {metrics['coverage_acc']:.1f}%")

        return " | ".join(parts)


class SpectralLoss(nn.Module):
    """
    Frequency-aware loss to combat spectral bias in FNOs.

    Standard L2 loss prioritizes low-frequency errors (they dominate energy).
    This loss explicitly penalizes errors across the frequency spectrum,
    helping the model learn high-frequency features.

    Based on: "Fourier Neural Operators for Structural Dynamics Models:
    Challenges, Limitations and Advantages of Using a Spectrogram Loss"
    """

    def __init__(self, weight_high_freq: float = 2.0):
        """
        Args:
            weight_high_freq: Weight multiplier for high-frequency errors.
                             Higher values push the model to learn sharp features.
        """
        super().__init__()
        self.weight_high_freq = weight_high_freq

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target shape: (batch, n_times, nx, ny)
        batch_size = pred.shape[0]

        # Compute 2D FFT for each time step
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        # Magnitude of error in frequency domain
        spectral_error = torch.abs(pred_fft - target_fft)

        # Create frequency weighting mask (higher weight for high frequencies)
        # Shape: (nx, ny//2+1) for rfft2 output
        nx, ny_rfft = spectral_error.shape[-2], spectral_error.shape[-1]

        # Frequency indices (normalized to [0, 1])
        freq_x = torch.fft.fftfreq(nx, device=pred.device).abs()
        freq_y = torch.linspace(0, 0.5, ny_rfft, device=pred.device)

        # 2D frequency magnitude
        freq_x_grid, freq_y_grid = torch.meshgrid(freq_x, freq_y, indexing='ij')
        freq_magnitude = torch.sqrt(freq_x_grid**2 + freq_y_grid**2)

        # Weight: 1.0 for low freq, weight_high_freq for high freq
        # Linear interpolation based on frequency magnitude
        freq_weight = 1.0 + (self.weight_high_freq - 1.0) * (freq_magnitude / freq_magnitude.max())

        # Apply frequency weighting
        weighted_error = spectral_error * freq_weight

        # Normalize by target spectrum magnitude
        target_magnitude = torch.abs(target_fft) + 1e-8
        relative_spectral_error = weighted_error / target_magnitude

        return relative_spectral_error.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function: Relative L2 + Spectral Loss.

    This addresses the spectral bias problem in FNOs by:
    1. RelativeLpLoss: Scale-invariant spatial loss (good for PDEs)
    2. SpectralLoss: Frequency-aware loss (captures high-frequency features)
    """

    def __init__(self, spatial_weight: float = 0.7, spectral_weight: float = 0.3):
        super().__init__()
        self.spatial_loss = RelativeLpLoss(p=2)
        self.spectral_loss = SpectralLoss(weight_high_freq=2.0)
        self.spatial_weight = spatial_weight
        self.spectral_weight = spectral_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_spatial = self.spatial_loss(pred, target)
        l_spectral = self.spectral_loss(pred, target)
        return self.spatial_weight * l_spatial + self.spectral_weight * l_spectral


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

        # Learning rate scheduler: warmup + cosine annealing
        warmup_epochs = min(5, n_epochs // 10)
        self.warmup_epochs = warmup_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=n_epochs - warmup_epochs,
            eta_min=lr * 0.01  # Reduce to 1% of initial LR
        )

        # Combined loss: Relative L2 + Spectral (to combat spectral bias)
        # This helps the model learn both low and high frequency features
        self.criterion = CombinedLoss(spatial_weight=0.7, spectral_weight=0.3)

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
            # pred shape: (batch, n_times, nx, ny)

            # Match dimensions - trajectory is (batch, n_times, ny, nx)
            # Need to transpose spatial dims to match pred
            n_times = pred.shape[1]
            target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)
            # target now: (batch, n_times, nx, ny)

            loss = self.criterion(pred, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate and compute all metrics."""
        self.model.eval()
        total_loss = 0.0
        batch_metrics = []

        for batch in self.val_loader:
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)

            pred = self.model(params)
            # pred shape: (batch, n_times, nx, ny)

            # Transpose target to match pred
            n_times = pred.shape[1]
            target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)

            total_loss += self.criterion(pred, target).item()

            # Compute detailed metrics
            metrics = Metrics.compute_all(pred, target)
            batch_metrics.append(metrics)

        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = Metrics.aggregate(batch_metrics)

        return avg_loss, avg_metrics

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
        print("-" * 100)

        initial_lr = self.optimizer.param_groups[0]['lr']
        self.metrics_history = []

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            # Warmup: linearly increase LR
            if epoch <= self.warmup_epochs:
                warmup_factor = epoch / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = initial_lr * warmup_factor
            else:
                # Cosine annealing after warmup
                self.scheduler.step()

            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.metrics_history.append(val_metrics)

            # Check for improvement
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

            # Print main info
            print(
                f"Epoch {epoch:3d} | "
                f"Train: {train_loss:.4e} | "
                f"Val: {val_loss:.4e} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
                + status
            )
            # Print detailed metrics
            print(f"         → {Metrics.format(val_metrics)}")

            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {self.patience} epochs)")
                self.early_stop = True
                break

        print("-" * 100)
        print(f"Training finished!")
        print(f"Best validation loss: {self.best_val_loss:.4e}")
        if hasattr(self, 'best_metrics'):
            print(f"Best metrics: {Metrics.format(self.best_metrics)}")


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

    train_loader, val_loader = create_dataloaders(
        data_dir, batch_size=16, train_ratio=0.8, num_workers=0
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
