"""
Training script for SuperB-FNO proppant model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class CombinedLoss(nn.Module):
    def __init__(
        self,
        mse_weight=1.0,
        rel_weight=0.5,
        temporal_weight=0.1,
        tv_spatial_weight=0.05,
        tv_temporal_weight=0.01,
        noise_weight=0.1,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.rel_weight = rel_weight
        self.temporal_weight = temporal_weight
        self.tv_spatial_weight = tv_spatial_weight
        self.tv_temporal_weight = tv_temporal_weight
        self.noise_weight = noise_weight

    def tv_spatial(self, x):
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        return dx.abs().mean() + dy.abs().mean()

    def tv_temporal(self, x):
        if x.shape[1] <= 1:
            return torch.tensor(0.0, device=x.device)
        dt = x[:, 1:] - x[:, :-1]
        return dt.abs().mean()

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)

        diff_norm = torch.norm(pred - target, p=2)
        target_norm = torch.norm(target, p=2) + 1e-6
        rel_loss = diff_norm / target_norm

        temporal_loss = torch.tensor(0.0, device=pred.device)
        if pred.shape[1] > 1:
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            temporal_loss = F.mse_loss(pred_diff, target_diff)

        tv_s = self.tv_spatial(pred)
        tv_t = self.tv_temporal(pred)

        # Noise suppression: penalize high-frequency variations
        # Apply avg_pool2d to spatial dimensions (nx, ny) for each time step
        # pred shape: (B, T, nx, ny)
        B, T, nx, ny = pred.shape
        # Reshape to (B*T, 1, nx, ny) for 2D pooling
        pred_reshaped = pred.contiguous().view(B * T, 1, nx, ny)
        # Apply 2D average pooling with padding to maintain size
        pred_smooth = F.avg_pool2d(pred_reshaped, kernel_size=3, stride=1, padding=1)
        # Reshape back to (B, T, nx, ny)
        pred_smooth = pred_smooth.view(B, T, nx, ny)
        noise_loss = F.mse_loss(pred, pred_smooth)

        total = (
            self.mse_weight * mse_loss
            + self.rel_weight * rel_loss
            + self.temporal_weight * temporal_loss
            + self.tv_spatial_weight * tv_s
            + self.tv_temporal_weight * tv_t
            + self.noise_weight * noise_loss
        )
        return total


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute evaluation metrics."""
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

    return {
        'mae': mae,
        'rmse': rmse,
        'rel_l2': rel_l2,
        'max_err': max_err,
        'accuracy': accuracy,
        'r2': max(0, r2 * 100),  # R² as percentage
    }


class Trainer:
    """Training manager for SuperB-FNO model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        optimizer_cfg: dict = None,
        scheduler_cfg: dict = None,
        loss_cfg: dict = None,
        trainer_cfg: dict = None,
        finetuning_cfg: dict = None,
        n_epochs: int = 100,
        checkpoint_dir: Path = None,
        start_epoch: int = 0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch

        finetuning_cfg = finetuning_cfg or {}
        self.finetuning_enabled = finetuning_cfg.get('enabled', False)
        self.freeze_base = finetuning_cfg.get('freeze_base', True)
        self.save_separate = finetuning_cfg.get('save_separate', True)

        if self.finetuning_enabled and self.freeze_base:
            for name, param in self.model.named_parameters():
                if 'error_corrector' not in name:
                    param.requires_grad = False

        optimizer_cfg = optimizer_cfg or {}
        opt_type = optimizer_cfg.get('type', 'AdamW').lower()
        lr = optimizer_cfg.get('lr', 1e-3)
        if self.finetuning_enabled:
            lr = finetuning_cfg.get('corrector_lr', lr)
        weight_decay = optimizer_cfg.get('weight_decay', 1e-4)
        betas = tuple(optimizer_cfg.get('betas', [0.9, 0.999]))
        eps = optimizer_cfg.get('eps', 1e-8)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if opt_type == 'adamw':
            self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        elif opt_type == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

        scheduler_cfg = scheduler_cfg or {}
        sched_type = scheduler_cfg.get('type', 'cosine_with_warmup')
        warmup_epochs = scheduler_cfg.get('warmup_epochs', min(5, max(1, n_epochs // 10)))
        self.warmup_epochs = warmup_epochs
        self.scheduler = None

        if sched_type == 'cosine_with_warmup' and n_epochs > warmup_epochs:
            T_max = scheduler_cfg.get('T_max')
            if T_max is None:
                T_max = n_epochs - warmup_epochs
            eta_min = scheduler_cfg.get('eta_min', lr * 0.01)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif sched_type == 'step':
            step_size = scheduler_cfg.get('step_size', 30)
            gamma = scheduler_cfg.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        loss_cfg = loss_cfg or {}
        self.criterion = CombinedLoss(
            mse_weight=loss_cfg.get('mse_weight', 1.0),
            rel_weight=loss_cfg.get('rel_weight', 0.5),
            temporal_weight=loss_cfg.get('temporal_weight', 0.1),
            tv_spatial_weight=loss_cfg.get('tv_spatial_weight', 0.05),
            tv_temporal_weight=loss_cfg.get('tv_temporal_weight', 0.01),
            noise_weight=loss_cfg.get('noise_weight', 0.1)
        )

        trainer_cfg = trainer_cfg or {}
        self.patience = trainer_cfg.get('patience', 8)
        self.min_delta = trainer_cfg.get('min_delta', 1e-4)
        self.grad_clip = trainer_cfg.get('grad_clip', 1.0)

        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # Create subdirectories for different smoothing types
            (checkpoint_dir / 'classic').mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / 'adaptive').mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.early_stop = False

    def load_checkpoint(self, checkpoint_path: Path):
        """Загрузка чекпоинта для дообучения."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if self.save_separate and 'base_model_state_dict' in checkpoint:
            base_state_dict = checkpoint['base_model_state_dict']
            corrector_state_dict = checkpoint.get('error_corrector_state_dict', {})
        else:
            state_dict = checkpoint['model_state_dict']
            base_state_dict = {k: v for k, v in state_dict.items() if 'error_corrector' not in k}
            corrector_state_dict = {k: v for k, v in state_dict.items() if 'error_corrector' in k}

        model_state = self.model.state_dict()
        filtered_base = {}
        filtered_corrector = {}
        skipped_keys = []

        # Detect smoothing approach compatibility
        model_has_adaptive = 'adaptive_temporal_smooth.kernel' in model_state
        checkpoint_has_adaptive = 'adaptive_temporal_smooth.kernel' in base_state_dict
        checkpoint_has_old = 'spatial_smooth.kernel' in base_state_dict or 'lowpass' in str(base_state_dict.keys())

        for key, value in base_state_dict.items():
            if key in model_state:
                if model_state[key].shape != value.shape:
                    if 'buffer' in key.lower() or key.endswith('.kernel'):
                        skipped_keys.append(f"{key} (shape mismatch)")
                        continue
                    skipped_keys.append(f"{key} (shape mismatch)")
                    continue
                # Skip incompatible smoothing weights
                if model_has_adaptive and ('spatial_smooth' in key or 'temporal_smooth' in key or 'lowpass' in key):
                    skipped_keys.append(f"{key} (incompatible smoothing)")
                    continue
                if not model_has_adaptive and 'adaptive_temporal_smooth' in key:
                    skipped_keys.append(f"{key} (incompatible smoothing)")
                    continue
                filtered_base[key] = value
            else:
                # Skip incompatible smoothing weights even if not in model
                if model_has_adaptive and ('spatial_smooth' in key or 'temporal_smooth' in key or 'lowpass' in key):
                    skipped_keys.append(f"{key} (incompatible smoothing)")
                    continue
                if not model_has_adaptive and 'adaptive_temporal_smooth' in key:
                    skipped_keys.append(f"{key} (incompatible smoothing)")
                    continue
                skipped_keys.append(f"{key} (not in model)")

        for key, value in corrector_state_dict.items():
            if key in model_state:
                if model_state[key].shape != value.shape:
                    skipped_keys.append(f"{key} (shape mismatch)")
                    continue
                filtered_corrector[key] = value

        filtered_state_dict = {**filtered_base, **filtered_corrector}
        self.model.load_state_dict(filtered_state_dict, strict=False)

        if skipped_keys:
            print(f"Warning: Skipped {len(skipped_keys)} keys during checkpoint loading")

        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                print("Warning: Could not load optimizer state, using fresh optimizer")

        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_accuracy = checkpoint.get('metrics', {}).get('accuracy', 0.0)

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                print("Warning: Could not load scheduler state, using fresh scheduler")

        print(f"Resumed from epoch {self.start_epoch}, best_val_loss={self.best_val_loss:.4e}")

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(params)

            # Match dimensions: trajectory is (batch, n_times, ny, nx)
            # Model outputs (batch, n_times, nx, ny)
            n_times = min(pred.shape[1], trajectory.shape[1])
            pred = pred[:, :n_times, :, :]

            # First trim trajectory by time, then permute to match model output format
            # Data format: trajectory is (B, T, ny, nx), model outputs (B, T, nx, ny)
            trajectory_trimmed = trajectory[:, :n_times, :, :]
            target = trajectory_trimmed.permute(0, 1, 3, 2)  # (B, T, ny, nx) -> (B, T, nx, ny)

            # Final assert - shapes must match exactly
            assert pred.shape == target.shape, f"Shape mismatch: pred={pred.shape}, target={target.shape}"

            loss = self.criterion(pred, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
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

            # Match dimensions: trajectory is (batch, n_times, ny, nx)
            # Model outputs (batch, n_times, nx, ny)
            n_times = min(pred.shape[1], trajectory.shape[1])
            pred = pred[:, :n_times, :, :]

            # First trim trajectory by time, then permute to match model output format
            # Data format: trajectory is (B, T, ny, nx), model outputs (B, T, nx, ny)
            trajectory_trimmed = trajectory[:, :n_times, :, :]
            target = trajectory_trimmed.permute(0, 1, 3, 2)  # (B, T, ny, nx) -> (B, T, nx, ny)

            # Final assert - shapes must match exactly
            assert pred.shape == target.shape, f"Shape mismatch: pred={pred.shape}, target={target.shape}"

            loss = self.criterion(pred, target)
            total_loss += loss.item()

            metrics = compute_metrics(pred, target)
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

        # Determine subdirectory based on smoothing type
        smoothing_type = 'adaptive' if self.model.use_adaptive_smoothing else 'classic'
        save_dir = self.checkpoint_dir / smoothing_type
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.save_separate and self.finetuning_enabled:
            base_state_dict = {k: v for k, v in self.model.state_dict().items() if 'error_corrector' not in k}
            corrector_state_dict = {k: v for k, v in self.model.state_dict().items() if 'error_corrector' in k}
            checkpoint = {
                'epoch': epoch,
                'base_model_state_dict': base_state_dict,
                'error_corrector_state_dict': corrector_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'metrics': metrics,
            }
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'metrics': metrics,
            }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, save_dir / 'last.pt')
        if is_best:
            torch.save(checkpoint, save_dir / 'best.pt')

    def train(self, n_epochs: int = None):
        if n_epochs is None:
            n_epochs = self.n_epochs

        print(f"Training on {self.device}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
        if self.finetuning_enabled:
            print(f"Finetuning: Only error_corrector is trainable")
        print(f"Scheduler: {self.warmup_epochs} warmup epochs + cosine annealing")
        print(f"Early stopping: patience={self.patience}, min_delta={self.min_delta}")
        print("-" * 100)
        print(f"{'Epoch':>5} | {'Train':>10} | {'Val':>10} | {'MAE':>8} | {'Acc%':>6} | {'R²%':>6} | {'LR':>9} | {'Time':>6} | Status")
        print("-" * 100)

        initial_lr = self.optimizer.param_groups[0]['lr']

        for epoch in range(self.start_epoch + 1, self.start_epoch + n_epochs + 1):
            t0 = time.time()

            # Warmup
            if epoch <= self.warmup_epochs:
                warmup_factor = epoch / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = initial_lr * warmup_factor

            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()

            if epoch > self.warmup_epochs and self.scheduler is not None:
                self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Check for improvement
            is_best = val_loss < (self.best_val_loss - self.min_delta)
            if is_best:
                self.best_val_loss = val_loss
                self.best_accuracy = metrics['accuracy']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best, metrics)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            status = ""
            if is_best:
                status = "[BEST]"
            elif self.patience_counter > 0:
                status = f"[{self.patience_counter}/{self.patience}]"

            print(
                f"{epoch:5d} | "
                f"{train_loss:10.4e} | "
                f"{val_loss:10.4e} | "
                f"{metrics['mae']:8.4f} | "
                f"{metrics['accuracy']:6.1f} | "
                f"{metrics['r2']:6.1f} | "
                f"{lr:9.2e} | "
                f"{elapsed:5.1f}s | "
                f"{status}"
            )

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {self.patience} epochs)")
                self.early_stop = True
                break

        print("-" * 100)
        print(f"Training finished!")
        print(f"Best validation loss: {self.best_val_loss:.4e}")
        print(f"Best accuracy: {self.best_accuracy:.1f}%")


def main(config_path: Path = None, resume_from: Path = None):
    """Main training entry point."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'processed'
    checkpoint_dir = project_root / 'checkpoints'

    if config_path is None:
        config_path = project_root / 'configs' / 'default.json'
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})
    optimizer_cfg = train_cfg.get('optimizer', {})
    scheduler_cfg = train_cfg.get('scheduler', {})
    loss_cfg = train_cfg.get('loss', {})
    trainer_cfg = train_cfg.get('trainer', {})
    finetuning_cfg = train_cfg.get('finetuning', {})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if finetuning_cfg.get('enabled', False):
        model_cfg = model_cfg.copy()
        model_cfg['use_error_corrector'] = True

    if not (data_dir / 'metadata.json').exists():
        print(f"No data found in {data_dir}")
        print("Run: python app.py --generate --samples 500")
        return

    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    batch_size = train_cfg.get('batch_size', 8)
    try:
        nx = int(metadata.get('nx', 0))
        ny = int(metadata.get('ny', 0))
        n_times_meta = int(metadata.get('n_times', 0))
        if nx * ny * max(n_times_meta, 1) >= 2_000_000:
            batch_size = min(batch_size, 4)
        if nx * ny * max(n_times_meta, 1) >= 4_000_000:
            batch_size = min(batch_size, 2)
    except Exception:
        batch_size = min(batch_size, 4)

    train_ratio = train_cfg.get('train_ratio', 0.8)
    max_samples = train_cfg.get('max_samples', None)
    train_loader, val_loader = create_dataloaders(
        data_dir, batch_size=batch_size, train_ratio=train_ratio, num_workers=0,
        config_path=config_path, max_samples=max_samples
    )

    sample = next(iter(train_loader))
    n_times = sample['trajectory'].shape[1]
    n_params = sample['params'].shape[1]

    n_epochs = train_cfg.get('n_epochs', 100)
    lr = optimizer_cfg.get('lr', 1e-3)
    patience = trainer_cfg.get('patience', 8)

    finetuning_enabled = finetuning_cfg.get('enabled', False)
    if finetuning_enabled:
        print(f"Finetuning mode: enabled")
        print(f"  - Freeze base model: {finetuning_cfg.get('freeze_base', True)}")
        print(f"  - Save separate: {finetuning_cfg.get('save_separate', True)}")
        print(f"  - Corrector LR: {finetuning_cfg.get('corrector_lr', lr)}")

    print(f"Training for {n_epochs} epochs (lr={lr}, patience={patience})...")
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Grid: {metadata['nx']}x{metadata['ny']}, {n_times} time steps, {n_params} params")

    # Check if resuming from checkpoint and detect model parameters BEFORE creating model
    resume_path = resume_from or train_cfg.get('resume_from')
    if resume_path:
        if isinstance(resume_path, str):
            resume_path = Path(resume_path)
        # Все относительные пути разрешаются относительно папки checkpoints
        if not resume_path.is_absolute():
            # Try adaptive first, then classic, then root
            adaptive_path = checkpoint_dir / 'adaptive' / resume_path
            classic_path = checkpoint_dir / 'classic' / resume_path
            root_path = checkpoint_dir / resume_path

            if adaptive_path.exists():
                resume_path = adaptive_path
            elif classic_path.exists():
                resume_path = classic_path
            elif root_path.exists():
                resume_path = root_path
            else:
                resume_path = checkpoint_dir / resume_path

        if not resume_path.exists():
            available = []
            if checkpoint_dir.exists():
                available.extend(checkpoint_dir.glob('*.pt'))
                available.extend((checkpoint_dir / 'adaptive').glob('*.pt'))
                available.extend((checkpoint_dir / 'classic').glob('*.pt'))
            raise FileNotFoundError(
                f"Checkpoint not found: {resume_path}\n"
                f"Available checkpoints: {[p.name for p in available]}"
            )

        # Load checkpoint to detect model parameters
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if 'base_model_state_dict' in checkpoint:
            state_dict = checkpoint['base_model_state_dict']
        else:
            state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Detect model parameters from checkpoint
        checkpoint_width = None
        checkpoint_modes1 = None
        checkpoint_modes2 = None
        checkpoint_use_adaptive = None

        if 'lift.0.weight' in state_dict:
            checkpoint_width = state_dict['lift.0.weight'].shape[0]
        elif 'fno_layers.0.local_conv.weight' in state_dict:
            checkpoint_width = state_dict['fno_layers.0.local_conv.weight'].shape[0]

        if 'fno_layers.0.spectral_conv.weights1' in state_dict:
            checkpoint_modes1 = state_dict['fno_layers.0.spectral_conv.weights1'].shape[2]
            checkpoint_modes2 = state_dict['fno_layers.0.spectral_conv.weights1'].shape[3]

        has_adaptive = 'adaptive_temporal_smooth.kernel' in state_dict
        has_old_smoothing = 'spatial_smooth.kernel' in state_dict or any('lowpass' in k for k in state_dict.keys())
        checkpoint_use_adaptive = has_adaptive and not has_old_smoothing

        # Update model_cfg with checkpoint parameters
        if checkpoint_width is not None:
            model_cfg['width'] = checkpoint_width
            print(f"  Using width={checkpoint_width} from checkpoint")
        if checkpoint_modes1 is not None:
            model_cfg['modes1'] = checkpoint_modes1
            print(f"  Using modes1={checkpoint_modes1} from checkpoint")
        if checkpoint_modes2 is not None:
            model_cfg['modes2'] = checkpoint_modes2
            print(f"  Using modes2={checkpoint_modes2} from checkpoint")
        if checkpoint_use_adaptive is not None:
            model_cfg['use_adaptive_smoothing'] = checkpoint_use_adaptive
            print(f"  Using use_adaptive_smoothing={checkpoint_use_adaptive} from checkpoint")

    model = create_model(
        nx=metadata['nx'],
        ny=metadata['ny'],
        n_times=n_times,
        n_params=n_params,
        device=device,
        model_cfg=model_cfg
    )

    finetuning_cfg = train_cfg.get('finetuning', {})

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        loss_cfg=loss_cfg,
        trainer_cfg=trainer_cfg,
        finetuning_cfg=finetuning_cfg,
        n_epochs=n_epochs,
        checkpoint_dir=checkpoint_dir
    )

    if resume_path:
        trainer.load_checkpoint(resume_path)

    trainer.train()


if __name__ == '__main__':
    main()
