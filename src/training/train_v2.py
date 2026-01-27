"""
Enhanced training script for FNO v2 proppant model.

Improvements:
1. Multi-GPU support (DataParallel / DistributedDataParallel)
2. Large batch size (32-64) with gradient accumulation
3. Advanced loss function (L2 + Spectral + H1 + Temporal)
4. Learning rate warmup + cosine annealing
5. Gradient clipping
6. EMA (Exponential Moving Average) for stable inference
7. Multiple metrics tracking
8. SpecBoost training (optional)
9. Automatic GPU detection and scaling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import os
import math
from pathlib import Path
import json
import time
import copy
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


def get_gpu_info():
    """Get information about available GPUs."""
    if not torch.cuda.is_available():
        return {
            'available': False,
            'count': 0,
            'devices': [],
            'total_memory_gb': 0,
            'free_memory_gb': 0,
        }

    count = torch.cuda.device_count()
    devices = []
    total_memory = 0
    min_free_memory = float('inf')

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)

        # Get free memory
        torch.cuda.set_device(i)
        free_memory = torch.cuda.mem_get_info()[0] / (1024**3)
        min_free_memory = min(min_free_memory, free_memory)

        devices.append({
            'id': i,
            'name': props.name,
            'memory_gb': memory_gb,
            'free_memory_gb': free_memory,
            'compute_capability': f"{props.major}.{props.minor}",
        })
        total_memory += memory_gb

    torch.cuda.set_device(0)  # Reset to first GPU

    return {
        'available': True,
        'count': count,
        'devices': devices,
        'total_memory_gb': total_memory,
        'min_free_memory_gb': min_free_memory if min_free_memory != float('inf') else 0,
    }


def estimate_memory_per_gpu(model_size: str, use_specboost: bool, batch_size: int, n_gpus: int = 1) -> float:
    """
    Estimate GPU memory usage in GB per GPU.

    With DataParallel, the model is replicated on each GPU, but gradients
    are averaged. So each GPU needs:
    - Full model weights
    - Gradients for batch_size / n_gpus samples
    - Activations for batch_size / n_gpus samples
    """
    # Base model parameters (millions)
    model_params = {
        'small': 5,
        'medium': 37,
        'large': 156,
        'xlarge': 438,
    }

    params_m = model_params.get(model_size, 156)
    if use_specboost:
        params_m = int(params_m * 1.4)  # SpecBoost adds ~40% more params

    # Memory estimation (conservative):
    # - Model weights: params * 4 bytes (float32)
    # - Gradients: params * 4 bytes
    # - Optimizer states (AdamW): params * 8 bytes (2 states: m and v)
    # - Model copy overhead during forward: params * 4 bytes (temporary)
    # - Activations: ~200MB per sample (FNO has large intermediate tensors)

    # Total: params * 20 bytes + activations
    model_memory_gb = params_m * 20 / 1000  # 20 bytes per param

    # With DataParallel, batch is split across GPUs
    samples_per_gpu = max(1, batch_size // n_gpus)
    activation_memory_gb = samples_per_gpu * 0.2  # ~200MB per sample (conservative)

    # CUDA overhead and fragmentation (20% buffer)
    total_gb = (model_memory_gb + activation_memory_gb) * 1.2

    return total_gb


def recommend_batch_size(model_size: str, use_specboost: bool, gpu_memory_gb: float, n_gpus: int) -> int:
    """
    Recommend optimal batch size based on available memory.

    Returns the largest batch size that should fit in memory with headroom.
    """
    # Leave more headroom for larger models (CUDA overhead, fragmentation)
    headroom = {'small': 4.0, 'medium': 5.0, 'large': 6.0, 'xlarge': 8.0}
    available_per_gpu = gpu_memory_gb - headroom.get(model_size, 6.0)

    # Model memory (fixed cost) - using conservative estimate
    model_params = {'small': 5, 'medium': 37, 'large': 156, 'xlarge': 438}
    params_m = model_params.get(model_size, 156)
    if use_specboost:
        params_m = int(params_m * 1.4)

    # 20 bytes per param: weights + grads + optimizer states + overhead
    model_memory_gb = params_m * 20 / 1000

    # Available for activations
    activation_budget = available_per_gpu - model_memory_gb

    if activation_budget <= 0:
        return 1  # Minimum, model barely fits

    # Memory per sample depends on model size (larger models = more activations)
    # Conservative: 100MB for small, 200MB for large, 400MB for xlarge
    mem_per_sample = {'small': 0.1, 'medium': 0.15, 'large': 0.25, 'xlarge': 0.4}
    if use_specboost:
        mem_per_sample = {k: v * 1.3 for k, v in mem_per_sample.items()}

    samples_per_gpu = int(activation_budget / mem_per_sample.get(model_size, 0.2))
    total_batch = samples_per_gpu * n_gpus

    # Cap based on model size (larger models = smaller max batch)
    max_batch = {'small': 128, 'medium': 64, 'large': 32, 'xlarge': 16}
    max_allowed = max_batch.get(model_size, 32)
    if use_specboost:
        max_allowed = max(4, max_allowed // 2)

    total_batch = max(1, min(max_allowed, total_batch))

    # Round down to power of 2 for efficiency
    for bs in [128, 64, 32, 16, 8, 4, 2, 1]:
        if bs <= total_batch:
            return bs

    return 1


def check_model_fits(model_size: str, use_specboost: bool, gpu_memory_gb: float) -> tuple:
    """
    Check if the model can fit on a single GPU (required for DataParallel).

    Returns:
        (fits: bool, model_memory_gb: float, message: str)
    """
    model_params = {'small': 5, 'medium': 37, 'large': 156, 'xlarge': 438}
    params_m = model_params.get(model_size, 156)
    if use_specboost:
        params_m = int(params_m * 1.4)

    # Minimum memory needed: weights + grads + optimizer + small batch
    # 20 bytes per param + 1GB for minimal activations
    min_memory_gb = params_m * 20 / 1000 + 1.0

    if min_memory_gb > gpu_memory_gb:
        return (
            False,
            min_memory_gb,
            f"Model requires ~{min_memory_gb:.1f}GB but GPU has {gpu_memory_gb:.1f}GB. "
            f"Try --model {get_smaller_model(model_size)} or remove --specboost"
        )

    return (True, min_memory_gb, "OK")


def can_use_data_parallel(model_size: str, use_specboost: bool, gpu_memory_gb: float) -> tuple:
    """
    Check if DataParallel can be safely used.

    DataParallel broadcasts model parameters to all GPUs, which requires
    significant memory overhead. For very large models (>300M params),
    the broadcast phase can cause OOM.

    Returns:
        (can_use: bool, reason: str)
    """
    model_params = {'small': 5, 'medium': 37, 'large': 156, 'xlarge': 438}
    params_m = model_params.get(model_size, 156)
    if use_specboost:
        params_m = int(params_m * 1.4)

    # DataParallel overhead during broadcast: need ~2x model weights in memory
    # (original + broadcast buffer)
    model_weights_gb = params_m * 4 / 1000  # 4 bytes per float32 param
    dataparallel_overhead_gb = model_weights_gb * 2

    # Need at least 8GB free after model for activations
    required_gb = dataparallel_overhead_gb + 8.0

    if required_gb > gpu_memory_gb:
        return (
            False,
            f"DataParallel requires ~{required_gb:.1f}GB/GPU but only {gpu_memory_gb:.1f}GB available. "
            f"Using single GPU mode with gradient checkpointing."
        )

    # Also check parameter count - very large models have issues
    if params_m > 400:  # > 400M params
        return (
            False,
            f"Model has {params_m}M params - too large for efficient DataParallel. "
            f"Using single GPU mode with gradient checkpointing."
        )

    return (True, "OK")


def get_smaller_model(model_size: str) -> str:
    """Get the next smaller model size."""
    order = ['xlarge', 'large', 'medium', 'small']
    try:
        idx = order.index(model_size)
        return order[min(idx + 1, len(order) - 1)]
    except ValueError:
        return 'small'


def setup_distributed(rank, world_size):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

from src.model.fno_v2 import (
    create_enhanced_model,
    CombinedLossV2,
    EnhancedFNOProppant,
    SpecBoostFNO
)
from src.training.dataset import ProppantDataset, create_dataloaders
from src.training.train import Metrics  # Reuse metrics from train.py


class EMA:
    """Exponential Moving Average for model weights (supports DataParallel)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        # Handle DataParallel wrapper
        self.model = model.module if hasattr(model, 'module') else model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class EnhancedTrainer:
    """
    Enhanced training manager for FNO v2.

    Features:
    - Multi-GPU support (DataParallel)
    - Large batch training with gradient accumulation
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
        multi_gpu: bool = True,  # Enable multi-GPU by default
    ):
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.multi_gpu = multi_gpu and self.n_gpus > 1

        # Move model to device first
        self.model = model.to(device)

        # Wrap with DataParallel if multiple GPUs
        if self.multi_gpu:
            print(f"Using DataParallel with {self.n_gpus} GPUs")
            self.model = DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_epochs = n_epochs
        # AMP doesn't support complex numbers (used in FFT), so disable it for FNO
        # See: https://github.com/pytorch/pytorch/issues/48108
        self.use_amp = False  # Disabled due to ComplexFloat incompatibility
        if use_amp:
            print("Note: AMP disabled (FFT uses complex numbers which are incompatible)")
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

        # Handle DataParallel: save the underlying model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'n_gpus': self.n_gpus,
        }

        torch.save(checkpoint, self.checkpoint_dir / 'last_v2.pt')
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_v2.pt')

        self.ema.restore()

    def train(self):
        # Get model without DataParallel wrapper for param count
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        n_params = sum(p.numel() for p in base_model.parameters())

        print("=" * 100)
        print("Enhanced FNO v2 Training")
        print("=" * 100)
        print(f"Device: {self.device}")
        if self.multi_gpu:
            print(f"Multi-GPU: {self.n_gpus} GPUs (DataParallel)")
            gpu_info = get_gpu_info()
            for gpu in gpu_info['devices']:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
            print(f"  Total VRAM: {gpu_info['total_memory_gb']:.1f} GB")
        print(f"Parameters: {n_params:,}")
        print(f"Mixed Precision: {self.use_amp} (disabled for FFT compatibility)")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps}x")
        effective_batch = self.train_loader.batch_size * self.gradient_accumulation_steps
        if self.multi_gpu:
            effective_batch *= self.n_gpus
            print(f"Effective Batch Size: {self.train_loader.batch_size} x {self.gradient_accumulation_steps} x {self.n_gpus} GPUs = {effective_batch}")
        else:
            print(f"Effective Batch Size: {effective_batch}")
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


def main(
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 0,    # 0 = auto-detect optimal batch size
    model_size: str = 'large',
    use_specboost: bool = False,
    patience: int = 50,
):
    """Main training entry point for FNO v2."""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ===== GPU Detection =====
    gpu_info = get_gpu_info()
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)

    if gpu_info['available']:
        print(f"Found {gpu_info['count']} GPU(s):")
        for gpu in gpu_info['devices']:
            print(f"  [{gpu['id']}] {gpu['name']} - {gpu['memory_gb']:.1f} GB "
                  f"(free: {gpu['free_memory_gb']:.1f} GB, CC {gpu['compute_capability']})")
        print(f"Total VRAM: {gpu_info['total_memory_gb']:.1f} GB")

        # Get minimum GPU memory (for DataParallel, all GPUs need to fit the model)
        min_gpu_memory = min(g['memory_gb'] for g in gpu_info['devices'])

        # ===== Check if model fits =====
        fits, model_mem, msg = check_model_fits(model_size, use_specboost, min_gpu_memory)
        if not fits:
            print("\n" + "!" * 60)
            print("ERROR: Model too large for available GPU memory!")
            print(msg)
            print("!" * 60)
            return

        print(f"\nModel memory estimate: ~{model_mem:.1f} GB per GPU")

        # ===== Auto-detect or validate batch size =====
        recommended_batch = recommend_batch_size(
            model_size, use_specboost, min_gpu_memory, gpu_info['count']
        )

        if batch_size <= 0:
            # Auto-detect: use recommended
            batch_size = recommended_batch
            print(f"Auto-detected batch size: {batch_size}")
        else:
            # User specified: validate and warn if too large
            estimated_mem = estimate_memory_per_gpu(
                model_size, use_specboost, batch_size, gpu_info['count']
            )
            if estimated_mem > min_gpu_memory - 2.0:  # Leave 2GB safety margin
                print(f"\n⚠️  Warning: batch_size={batch_size} may cause OOM!")
                print(f"   Estimated memory: {estimated_mem:.1f} GB/GPU")
                print(f"   Available: {min_gpu_memory:.1f} GB/GPU")
                print(f"   Recommended: {recommended_batch}")
                print(f"   Auto-adjusting batch_size to {recommended_batch}")
                batch_size = recommended_batch

        # ===== Check if DataParallel is safe for this model size =====
        use_multi_gpu = False
        if gpu_info['count'] > 1:
            can_dp, dp_reason = can_use_data_parallel(model_size, use_specboost, min_gpu_memory)
            if can_dp:
                use_multi_gpu = True
                print(f"\nMulti-GPU mode: {gpu_info['count']} GPUs (DataParallel)")
                print(f"Batch size per GPU: ~{batch_size // gpu_info['count']}")
                print(f"Total batch per step: {batch_size}")
            else:
                use_multi_gpu = False
                print(f"\n⚠️  {dp_reason}")
                print("Training on single GPU with gradient checkpointing for memory efficiency.")
                # For single GPU with very large model, recommend smaller batch
                batch_size = recommend_batch_size(model_size, use_specboost, min_gpu_memory, 1)
                print(f"Adjusted batch size for single GPU: {batch_size}")

        # Clear CUDA cache before model creation
        torch.cuda.empty_cache()
        for i in range(gpu_info['count']):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

        # Initialize for single GPU case (when count == 1)
        if not use_multi_gpu and gpu_info['count'] == 1:
            print("\nSingle GPU mode")

    else:
        print("No GPU available, using CPU")
        if batch_size <= 0:
            batch_size = 8  # Small default for CPU
        use_multi_gpu = False

    print("=" * 60)

    # Check if data exists
    if not (data_dir / 'metadata.json').exists():
        print(f"No data found in {data_dir}")
        print("Run: python app.py --generate --samples 500")
        return

    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    # Create dataloaders with larger batch size
    # Use more workers for multi-GPU
    num_workers = min(8, 4 * gpu_info['count']) if gpu_info['available'] else 0
    train_loader, val_loader = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        train_ratio=0.8,
        num_workers=num_workers
    )

    # Get dimensions from data
    sample = next(iter(train_loader))
    n_times = sample['trajectory'].shape[1]
    n_params = sample['params'].shape[1]

    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Grid: {metadata['nx']}x{metadata['ny']}, {n_times} time steps, {n_params} params")
    print(f"Model size: {model_size}, SpecBoost: {use_specboost}")

    # Create model with gradient checkpointing for very large models
    use_gradient_checkpointing = (model_size == 'xlarge' or use_specboost) and not use_multi_gpu

    model = create_enhanced_model(
        nx=metadata['nx'],
        ny=metadata['ny'],
        n_times=n_times,
        n_params=n_params,
        device=device,
        model_size=model_size,
        use_specboost=use_specboost,
    )

    # Enable gradient checkpointing if needed
    if use_gradient_checkpointing:
        print("Enabling gradient checkpointing for memory efficiency...")
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        else:
            # Fallback: enable for any modules that support it
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True

    # Create trainer
    # Note: DataParallel splits batch across GPUs, it doesn't multiply
    # So effective batch = batch_size * grad_accum (NOT * n_gpus)
    # For single GPU with very large model, use more gradient accumulation
    if use_multi_gpu:
        grad_accum = 4
    else:
        # More accumulation for single GPU to maintain effective batch size
        grad_accum = 8 if model_size in ['xlarge', 'large'] else 4

    # Scale learning rate with effective batch size (linear scaling rule)
    effective_batch = batch_size * grad_accum
    scaled_lr = lr * (effective_batch / 128)  # Base LR is for batch=128

    print(f"Effective batch size: {batch_size} x {grad_accum} = {effective_batch}")
    print(f"Learning rate scaled: {lr} -> {scaled_lr:.6f} (linear scaling)")

    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=scaled_lr,
        n_epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        use_amp=False,  # Disabled for FFT compatibility
        gradient_accumulation_steps=grad_accum,
        multi_gpu=use_multi_gpu,
    )

    trainer.train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train FNO v2')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Batch size (0 = auto-detect based on GPU memory)')
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
