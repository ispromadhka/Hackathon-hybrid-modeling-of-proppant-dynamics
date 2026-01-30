"""
Training script for SuperB-FNO proppant model.
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


class CombinedLoss(nn.Module):
    """
    Combined loss for proppant transport:
    - MSE for accurate predictions
    - Relative L2 for scale-invariance
    - Temporal consistency for smooth evolution
    """

    def __init__(self, mse_weight: float = 1.0, rel_weight: float = 0.5, temporal_weight: float = 0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.rel_weight = rel_weight
        self.temporal_weight = temporal_weight

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

        optimizer_cfg = optimizer_cfg or {}
        opt_type = optimizer_cfg.get('type', 'AdamW').lower()
        lr = optimizer_cfg.get('lr', 1e-3)
        weight_decay = optimizer_cfg.get('weight_decay', 1e-4)
        betas = tuple(optimizer_cfg.get('betas', [0.9, 0.999]))
        eps = optimizer_cfg.get('eps', 1e-8)

        if opt_type == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        elif opt_type == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        else:
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

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
            temporal_weight=loss_cfg.get('temporal_weight', 0.1)
        )

        trainer_cfg = trainer_cfg or {}
        self.patience = trainer_cfg.get('patience', 8)
        self.min_delta = trainer_cfg.get('min_delta', 1e-4)
        self.grad_clip = trainer_cfg.get('grad_clip', 1.0)

        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        state_dict = checkpoint['model_state_dict']

        # Фильтруем параметры и buffers, которые могут не совпадать из-за разных размеров
        model_state = self.model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []

        for key, value in state_dict.items():
            if key in model_state:
                # Проверяем совпадение размеров для buffers и параметров
                if model_state[key].shape != value.shape:
                    # Пропускаем buffers с несовпадающими размерами (будут пересозданы)
                    if 'buffer' in key.lower() or key.endswith('.kernel') or 'spatial_smooth' in key:
                        skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state[key].shape})")
                        continue
                    else:
                        # Для параметров с несовпадающими размерами тоже пропускаем
                        skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state[key].shape})")
                        continue
                filtered_state_dict[key] = value
            else:
                skipped_keys.append(f"{key} (not in model)")

        missing_keys = set(model_state.keys()) - set(filtered_state_dict.keys())
        # Удаляем buffers из missing_keys, так как они будут пересозданы при инициализации
        missing_keys = {k for k in missing_keys if not (
            k.endswith('.kernel') or 'spatial_smooth' in k or 'buffer' in k.lower()
        )}

        self.model.load_state_dict(filtered_state_dict, strict=False)

        if skipped_keys:
            print(f"Warning: Skipped {len(skipped_keys)} keys during checkpoint loading:")
            for key in skipped_keys[:5]:
                print(f"  - {key}")
            if len(skipped_keys) > 5:
                print(f"  ... and {len(skipped_keys) - 5} more")

        if missing_keys:
            print(f"Warning: {len(missing_keys)} model keys not found in checkpoint (will use random initialization)")

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_accuracy = checkpoint.get('metrics', {}).get('accuracy', 0.0)

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

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
            n_times = pred.shape[1]
            target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)

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

            n_times = pred.shape[1]
            target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)

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

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

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

    train_cfg = cfg.get('training', {})
    optimizer_cfg = train_cfg.get('optimizer', {})
    scheduler_cfg = train_cfg.get('scheduler', {})
    loss_cfg = train_cfg.get('loss', {})
    trainer_cfg = train_cfg.get('trainer', {})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    train_loader, val_loader = create_dataloaders(
        data_dir, batch_size=batch_size, train_ratio=train_ratio, num_workers=0, config_path=config_path
    )

    sample = next(iter(train_loader))
    n_times = sample['trajectory'].shape[1]
    n_params = sample['params'].shape[1]

    n_epochs = train_cfg.get('n_epochs', 100)
    lr = optimizer_cfg.get('lr', 1e-3)
    patience = trainer_cfg.get('patience', 8)

    print(f"Training for {n_epochs} epochs (lr={lr}, patience={patience})...")
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
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        loss_cfg=loss_cfg,
        trainer_cfg=trainer_cfg,
        n_epochs=n_epochs,
        checkpoint_dir=checkpoint_dir
    )

    resume_path = resume_from or train_cfg.get('resume_from')
    if resume_path:
        if isinstance(resume_path, str):
            resume_path = Path(resume_path)
        # Все относительные пути разрешаются относительно папки checkpoints
        if not resume_path.is_absolute():
            resume_path = checkpoint_dir / resume_path
        if not resume_path.exists():
            available = list(checkpoint_dir.glob('*.pt')) if checkpoint_dir.exists() else []
            raise FileNotFoundError(
                f"Checkpoint not found: {resume_path}\n"
                f"Available checkpoints in {checkpoint_dir}: {[p.name for p in available]}"
            )
        trainer.load_checkpoint(resume_path)

    trainer.train()


if __name__ == '__main__':
    main()
