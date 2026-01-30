"""
Training script for SuperB-FNO proppant model.

Features:
- Physics-informed loss (mass conservation, boundary conditions)
- Spectral loss for frequency-aware training
- High-Frequency Scaling (HFS) support
- Cosine annealing with warm restarts
- Separate learning rates for spectral components
- Multi-GPU support with DataParallel
- Mixed precision training (AMP) for faster training
- Optimized data loading with prefetching
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import json
import time
import math
import os
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


def get_device_info():
    """Get information about available compute devices."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cpu_count': os.cpu_count() or 1,
    }

    # Debug: Check why CUDA might not be available
    if not info['cuda_available']:
        print("DEBUG: CUDA not available. Checking reasons...")
        print(f"  torch.version.cuda: {torch.version.cuda}")
        print(f"  torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled if hasattr(torch.backends, 'cudnn') else 'N/A'}")
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

        # Try to get more info
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpus = result.stdout.strip().split('\n')
                print(f"  nvidia-smi found {len(gpus)} GPUs: {gpus}")
            else:
                print(f"  nvidia-smi error: {result.stderr}")
        except Exception as e:
            print(f"  nvidia-smi check failed: {e}")

    if info['cuda_available']:
        info['cuda_devices'] = [torch.cuda.get_device_name(i) for i in range(info['cuda_count'])]
        info['cuda_memory'] = [torch.cuda.get_device_properties(i).total_memory // (1024**3)
                               for i in range(info['cuda_count'])]

    return info


def setup_device(prefer_multi_gpu: bool = True):
    """
    Setup compute device with optimal configuration.

    Args:
        prefer_multi_gpu: Use DataParallel if multiple GPUs available

    Returns:
        device, use_amp, is_distributed
    """
    info = get_device_info()

    if info['cuda_available']:
        device = torch.device('cuda')
        use_amp = True  # Enable mixed precision on CUDA

        if info['cuda_count'] > 1 and prefer_multi_gpu:
            print(f"Multi-GPU: {info['cuda_count']} GPUs available")
            for i, (name, mem) in enumerate(zip(info['cuda_devices'], info['cuda_memory'])):
                print(f"  GPU {i}: {name} ({mem} GB)")
            is_distributed = True
        else:
            print(f"Single GPU: {info['cuda_devices'][0]}")
            is_distributed = False

    elif info['mps_available']:
        device = torch.device('mps')
        use_amp = False  # MPS doesn't support AMP well yet
        is_distributed = False
        print("Using Apple MPS (Metal)")

    else:
        device = torch.device('cpu')
        use_amp = False
        is_distributed = False
        print(f"Using CPU ({info['cpu_count']} cores)")

    return device, use_amp, is_distributed

from src.model.fno import create_model
from src.training.dataset import ProppantDataset, create_dataloaders


class SpectralLoss(nn.Module):
    """
    Spectral loss for frequency-aware training.

    Computes loss in Fourier domain to ensure both low and high
    frequency components are learned properly.
    """

    def __init__(self, weight_low: float = 1.0, weight_high: float = 2.0, cutoff: float = 0.3):
        super().__init__()
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.cutoff = cutoff

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # cuFFT doesn't support half precision for non-power-of-two sizes
        # Always compute FFT in float32
        pred_float = pred.float()
        target_float = target.float()

        # FFT of prediction and target
        pred_ft = torch.fft.rfft2(pred_float)
        target_ft = torch.fft.rfft2(target_float)

        # Magnitude spectra
        pred_mag = torch.abs(pred_ft)
        target_mag = torch.abs(target_ft)

        # Create frequency mask
        size1, size2 = pred_ft.shape[-2], pred_ft.shape[-1]
        freq1 = torch.fft.fftfreq(size1, device=pred.device)
        freq2 = torch.fft.rfftfreq(pred_float.shape[-1], device=pred.device)
        freq_dist = torch.sqrt(freq1[:, None]**2 + freq2[None, :]**2)
        freq_dist = freq_dist / (freq_dist.max() + 1e-8)

        low_mask = (freq_dist < self.cutoff).float()
        high_mask = 1.0 - low_mask

        # Separate losses for low and high frequencies
        diff_mag = (pred_mag - target_mag) ** 2
        low_loss = (diff_mag * low_mask).mean()
        high_loss = (diff_mag * high_mask).mean()

        return self.weight_low * low_loss + self.weight_high * high_loss


class MassConservationLoss(nn.Module):
    """
    Physics-informed loss for mass conservation.

    Penalizes violations of mass conservation law:
    - Total mass should only change due to inlet/outlet flux
    - Interior mass changes should be smooth

    Based on: "Conservation-preserved FNO" (arxiv 2505.24579)
    """

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute mass at each timestep
        pred_mass = pred.sum(dim=(-2, -1))  # (batch, n_times)
        target_mass = target.sum(dim=(-2, -1))

        # Mass should follow same pattern as target
        mass_error = torch.mean((pred_mass - target_mass) ** 2)

        # Penalize unphysical mass changes (should be monotonic or smooth)
        if pred.shape[1] > 1:
            pred_mass_diff = pred_mass[:, 1:] - pred_mass[:, :-1]
            target_mass_diff = target_mass[:, 1:] - target_mass[:, :-1]

            # Mass change should match target mass change
            mass_change_error = torch.mean((pred_mass_diff - target_mass_diff) ** 2)

            # Penalize sudden mass jumps
            mass_smoothness = torch.mean(torch.abs(pred_mass_diff[:, 1:] - pred_mass_diff[:, :-1]))
        else:
            mass_change_error = torch.tensor(0.0, device=pred.device)
            mass_smoothness = torch.tensor(0.0, device=pred.device)

        return self.weight * (mass_error + 0.5 * mass_change_error + 0.2 * mass_smoothness)


class BoundaryLoss(nn.Module):
    """
    Loss for boundary condition enforcement.

    Ensures:
    - No-flux at top/bottom boundaries (for settling proppant)
    - Correct inlet conditions
    """

    def __init__(self, weight: float = 0.3):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Top/bottom boundary error (should match target)
        top_error = torch.mean((pred[:, :, :, 0] - target[:, :, :, 0]) ** 2)
        bottom_error = torch.mean((pred[:, :, :, -1] - target[:, :, :, -1]) ** 2)

        # Left boundary (inlet) - important for correct injection
        left_error = torch.mean((pred[:, :, 0, :] - target[:, :, 0, :]) ** 2)

        return self.weight * (top_error + bottom_error + 2.0 * left_error)


class CombinedLoss(nn.Module):
    """
    Combined physics-informed loss for proppant transport:
    - MSE for accurate predictions
    - Relative L2 for scale-invariance
    - Temporal consistency for smooth evolution
    - Spectral loss for frequency-aware training
    - Mass conservation loss for physics consistency
    - Boundary loss for correct boundary conditions
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        rel_weight: float = 0.5,
        temporal_weight: float = 0.1,
        spectral_weight: float = 0.3,
        mass_weight: float = 0.5,
        boundary_weight: float = 0.3,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.rel_weight = rel_weight
        self.temporal_weight = temporal_weight

        # Physics-informed components
        self.spectral_loss = SpectralLoss(weight_low=1.0, weight_high=2.0) if spectral_weight > 0 else None
        self.spectral_weight = spectral_weight

        self.mass_loss = MassConservationLoss() if mass_weight > 0 else None
        self.mass_weight = mass_weight

        self.boundary_loss = BoundaryLoss() if boundary_weight > 0 else None
        self.boundary_weight = boundary_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MSE loss
        mse_loss = torch.mean((pred - target) ** 2)

        # Relative L2 loss (with epsilon for stability)
        diff_norm = torch.norm(pred - target, p=2)
        target_norm = torch.norm(target, p=2) + 1e-6
        rel_loss = diff_norm / target_norm

        # Temporal consistency: penalize large jumps between frames
        if pred.shape[1] > 1:
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            temporal_loss = torch.mean((pred_diff - target_diff) ** 2)
        else:
            temporal_loss = torch.tensor(0.0, device=pred.device)

        total = self.mse_weight * mse_loss + self.rel_weight * rel_loss + self.temporal_weight * temporal_loss

        # Add spectral loss
        if self.spectral_loss is not None and self.spectral_weight > 0:
            total = total + self.spectral_weight * self.spectral_loss(pred, target)

        # Add mass conservation loss
        if self.mass_loss is not None and self.mass_weight > 0:
            total = total + self.mass_weight * self.mass_loss(pred, target)

        # Add boundary loss
        if self.boundary_loss is not None and self.boundary_weight > 0:
            total = total + self.boundary_weight * self.boundary_loss(pred, target)

        return total


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute comprehensive evaluation metrics including physics-based metrics."""
    with torch.no_grad():
        # MAE (Mean Absolute Error)
        mae = torch.mean(torch.abs(pred - target)).item()

        # RMSE
        rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()

        # Relative L2 error
        rel_l2 = (torch.norm(pred - target) / (torch.norm(target) + 1e-6)).item()

        # Max error
        max_err = torch.max(torch.abs(pred - target)).item()

        # Accuracy: % of predictions within 5% of target range [0, 1]
        accuracy = (torch.abs(pred - target) < 0.05).float().mean().item() * 100

        # R² score
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - target.mean()) ** 2) + 1e-6
        r2 = (1 - ss_res / ss_tot).item()

        # Mass conservation metric
        pred_mass = pred.sum(dim=(-2, -1))
        target_mass = target.sum(dim=(-2, -1))
        mass_error = torch.mean(torch.abs(pred_mass - target_mass) / (target_mass.abs() + 1e-6)).item() * 100

        # Spectral error (high-frequency component)
        # Use float32 for FFT (cuFFT doesn't support half precision for non-power-of-two)
        pred_ft = torch.fft.rfft2(pred.float())
        target_ft = torch.fft.rfft2(target.float())
        spectral_error = torch.mean(torch.abs(torch.abs(pred_ft) - torch.abs(target_ft))).item()

        # Temporal smoothness (how well dynamics are captured)
        if pred.shape[1] > 1:
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            temporal_error = torch.mean(torch.abs(pred_diff - target_diff)).item()
        else:
            temporal_error = 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'rel_l2': rel_l2,
        'max_err': max_err,
        'accuracy': accuracy,
        'r2': max(0, r2 * 100),
        'mass_err': mass_error,
        'spectral_err': spectral_error,
        'temporal_err': temporal_error,
    }


class Trainer:
    """
    Training manager for SuperB-FNO model with physics-informed losses.

    Features:
    - Multi-GPU support with DataParallel
    - Mixed precision training (AMP)
    - Gradient accumulation for large batches
    - Physics-informed losses
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        lr: float = 1e-3,
        n_epochs: int = 100,
        checkpoint_dir: Path = None,
        patience: int = 15,
        min_delta: float = 1e-4,
        use_physics_loss: bool = True,
        use_amp: bool = True,
        use_multi_gpu: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        self.device = device
        self.n_epochs = n_epochs
        self.use_physics_loss = use_physics_loss
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Setup AMP (Automatic Mixed Precision)
        self.use_amp = use_amp and device != 'cpu' and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Setup multi-GPU
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        if self.use_multi_gpu:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(model)
        else:
            self.model = model
        self.model = self.model.to(device)

        # Store base model reference for checkpointing
        self.base_model = model if not self.use_multi_gpu else self.model.module

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Use AdamW with separate parameter groups for better optimization
        # Spectral weights need lower LR to avoid instability
        spectral_params = []
        other_params = []
        for name, param in self.base_model.named_parameters():
            if 'weights1' in name or 'weights2' in name or 'freq_weight' in name:
                spectral_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': other_params, 'lr': lr, 'weight_decay': 1e-4},
            {'params': spectral_params, 'lr': lr * 0.5, 'weight_decay': 1e-5},  # Lower LR for spectral
        ])

        warmup_epochs = min(10, max(2, n_epochs // 8))
        self.warmup_epochs = warmup_epochs
        self.scheduler = None
        if n_epochs > warmup_epochs:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(5, (n_epochs - warmup_epochs) // 3),
                T_mult=2,
                eta_min=lr * 0.001
            )

        # Physics-informed loss with all components
        if use_physics_loss:
            self.criterion = CombinedLoss(
                mse_weight=1.0,
                rel_weight=0.5,
                temporal_weight=0.2,
                spectral_weight=0.3,
                mass_weight=0.5,
                boundary_weight=0.3,
            )
        else:
            self.criterion = CombinedLoss(
                mse_weight=1.0,
                rel_weight=0.5,
                temporal_weight=0.1,
                spectral_weight=0.0,
                mass_weight=0.0,
                boundary_weight=0.0,
            )

        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_mass_err = float('inf')

        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.early_stop = False

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            params = batch['params'].to(self.device, non_blocking=True)
            trajectory = batch['trajectory'].to(self.device, non_blocking=True)

            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    pred = self.model(params)
                    n_times = pred.shape[1]
                    target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)
                    loss = self.criterion(pred, target)
                    loss = loss / self.gradient_accumulation_steps

                # Scaled backward pass
                self.scaler.scale(loss).backward()

                # Gradient accumulation step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                pred = self.model(params)
                n_times = pred.shape[1]
                target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)
                loss = self.criterion(pred, target)
                loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        all_metrics = []

        for batch in self.val_loader:
            params = batch['params'].to(self.device, non_blocking=True)
            trajectory = batch['trajectory'].to(self.device, non_blocking=True)

            # Mixed precision inference
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

            # Compute metrics in float32 for accuracy
            metrics = compute_metrics(pred.float(), target.float())
            all_metrics.append(metrics)

        avg_loss = total_loss / len(self.val_loader)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        return avg_loss, avg_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False, metrics: dict = None):
        if not self.checkpoint_dir:
            return

        # Save the unwrapped model (not DataParallel wrapper)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.base_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'use_amp': self.use_amp,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / 'last.pt')
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')

    def train(self, n_epochs: int = None):
        if n_epochs is None:
            n_epochs = self.n_epochs

        print(f"Training on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.base_model.parameters()):,}")
        print(f"Multi-GPU: {self.use_multi_gpu} | Mixed Precision (AMP): {self.use_amp}")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps} steps")
        print(f"Physics-informed loss: {self.use_physics_loss}")
        print(f"Scheduler: {self.warmup_epochs} warmup epochs + cosine annealing with restarts")
        print(f"Early stopping: patience={self.patience}, min_delta={self.min_delta}")
        print("-" * 120)
        print(f"{'Epoch':>5} | {'Train':>10} | {'Val':>10} | {'MAE':>8} | {'Acc%':>6} | {'R²%':>6} | {'Mass%':>6} | {'LR':>9} | {'Time':>6} | Status")
        print("-" * 120)

        initial_lrs = [pg['lr'] for pg in self.optimizer.param_groups]

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            # Warmup with smooth ramp
            if epoch <= self.warmup_epochs:
                # Cosine warmup for smoother start
                warmup_factor = 0.5 * (1 - math.cos(math.pi * epoch / self.warmup_epochs))
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = initial_lrs[i] * max(0.1, warmup_factor)

            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()

            if epoch > self.warmup_epochs and self.scheduler is not None:
                self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Check for improvement (consider both loss and accuracy)
            is_best = val_loss < (self.best_val_loss - self.min_delta)
            is_best_mass = metrics.get('mass_err', float('inf')) < self.best_mass_err

            if is_best:
                self.best_val_loss = val_loss
                self.best_accuracy = metrics['accuracy']
                self.best_mass_err = metrics.get('mass_err', float('inf'))
                self.patience_counter = 0
            elif is_best_mass:
                self.best_mass_err = metrics['mass_err']
                self.patience_counter = max(0, self.patience_counter - 1)
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best, metrics)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            status = ""
            if is_best:
                status = "[BEST]"
            elif is_best_mass:
                status = "[MASS]"
            elif self.patience_counter > 0:
                status = f"[{self.patience_counter}/{self.patience}]"

            mass_err = metrics.get('mass_err', 0.0)
            print(
                f"{epoch:5d} | "
                f"{train_loss:10.4e} | "
                f"{val_loss:10.4e} | "
                f"{metrics['mae']:8.4f} | "
                f"{metrics['accuracy']:6.1f} | "
                f"{metrics['r2']:6.1f} | "
                f"{mass_err:6.1f} | "
                f"{lr:9.2e} | "
                f"{elapsed:5.1f}s | "
                f"{status}"
            )

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {self.patience} epochs)")
                self.early_stop = True
                break

        print("-" * 120)
        print(f"Training finished!")
        print(f"Best validation loss: {self.best_val_loss:.4e}")
        print(f"Best accuracy: {self.best_accuracy:.1f}%")
        print(f"Best mass error: {self.best_mass_err:.1f}%")


def main(
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    use_physics_loss: bool = True,
    use_amp: bool = True,
    use_multi_gpu: bool = True,
    num_workers: int = 4,
    gradient_accumulation: int = 1,
):
    """
    Main training entry point with GPU parallelization.

    Args:
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        use_physics_loss: Whether to use physics-informed losses
        use_amp: Use mixed precision training (faster on GPU)
        use_multi_gpu: Use DataParallel for multiple GPUs
        num_workers: Number of data loading workers
        gradient_accumulation: Gradient accumulation steps (for larger effective batch size)
    """
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'

    # Setup device with optimal configuration
    device, auto_amp, is_distributed = setup_device(prefer_multi_gpu=use_multi_gpu)

    # Use auto-detected AMP setting if not explicitly disabled
    use_amp = use_amp and auto_amp

    if not (data_dir / 'metadata.json').exists():
        print(f"No data found in {data_dir}")
        print("Run: python app.py --generate --samples 500")
        return

    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    # Determine batch size based on available GPU memory and grid size
    batch_size = 8
    try:
        nx = int(metadata.get('nx', 0))
        ny = int(metadata.get('ny', 0))
        n_times_meta = int(metadata.get('n_times', 0))
        memory_per_sample = nx * ny * n_times_meta * 4 / (1024**3)  # GB per sample

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Use about 60% of GPU memory for data
            max_batch = max(1, int(gpu_memory * 0.6 / memory_per_sample))
            batch_size = min(16, max_batch)

            # With multi-GPU, can increase batch size
            if is_distributed:
                batch_size *= torch.cuda.device_count()
        else:
            if nx * ny * max(n_times_meta, 1) >= 2_000_000:
                batch_size = 4
            if nx * ny * max(n_times_meta, 1) >= 4_000_000:
                batch_size = 2

        print(f"Auto batch size: {batch_size} (memory/sample: {memory_per_sample:.3f} GB)")
    except Exception as e:
        print(f"Batch size detection failed: {e}, using default batch_size=4")
        batch_size = 4

    # Optimized data loading
    pin_memory = torch.cuda.is_available()
    train_loader, val_loader = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        train_ratio=0.8,
        num_workers=num_workers if torch.cuda.is_available() else 0
    )

    # Enable prefetching for faster data loading
    if pin_memory:
        train_loader.pin_memory = True
        val_loader.pin_memory = True

    sample = next(iter(train_loader))
    n_times = sample['trajectory'].shape[1]
    n_params = sample['params'].shape[1]

    print(f"Training for {epochs} epochs (lr={lr}, patience={patience})...")
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Grid: {metadata['nx']}x{metadata['ny']}, {n_times} time steps, {n_params} params")
    print(f"Batch size: {batch_size}, Workers: {num_workers}, Pin memory: {pin_memory}")

    model = create_model(
        nx=metadata['nx'],
        ny=metadata['ny'],
        n_times=n_times,
        n_params=n_params,
        device='cpu',  # Create on CPU first, then move to device in Trainer
        use_boundary_mask=use_physics_loss,
        use_mass_conservation=use_physics_loss,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        n_epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        use_physics_loss=use_physics_loss,
        use_amp=use_amp,
        use_multi_gpu=use_multi_gpu and is_distributed,
        gradient_accumulation_steps=gradient_accumulation,
    )

    trainer.train()


if __name__ == '__main__':
    main()
