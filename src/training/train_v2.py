"""
Enhanced training script for FNO v2 proppant model.

Supports:
1. FSDP (Fully Sharded Data Parallel) for very large models (>400M params)
2. DDP (Distributed Data Parallel) for medium models
3. Single GPU with gradient checkpointing for fallback
4. Automatic GPU detection and optimal strategy selection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import math
from pathlib import Path
import json
import time
from tqdm import tqdm
from functools import partial

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


# ============== GPU INFO ==============

def get_gpu_info():
    """Get information about available GPUs."""
    if not torch.cuda.is_available():
        return {
            'available': False,
            'count': 0,
            'devices': [],
            'total_memory_gb': 0,
            'min_free_memory_gb': 0,
        }

    count = torch.cuda.device_count()
    devices = []
    total_memory = 0
    min_free_memory = float('inf')

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)

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

    torch.cuda.set_device(0)

    return {
        'available': True,
        'count': count,
        'devices': devices,
        'total_memory_gb': total_memory,
        'min_free_memory_gb': min_free_memory if min_free_memory != float('inf') else 0,
    }


def get_model_params_millions(model_size: str, use_specboost: bool) -> int:
    """Get model parameter count in millions."""
    model_params = {'small': 5, 'medium': 37, 'large': 156, 'xlarge': 438}
    params_m = model_params.get(model_size, 156)
    if use_specboost:
        params_m = int(params_m * 1.4)
    return params_m


def recommend_strategy(model_size: str, use_specboost: bool, n_gpus: int, gpu_memory_gb: float) -> str:
    """
    Recommend training strategy based on model size and GPU configuration.

    Returns: 'fsdp', 'ddp', or 'single'
    """
    params_m = get_model_params_millions(model_size, use_specboost)

    # FSDP for very large models (>300M params) with multiple GPUs
    if params_m > 300 and n_gpus > 1:
        return 'fsdp'

    # DDP for medium-large models with multiple GPUs
    if n_gpus > 1 and params_m * 20 / 1000 < gpu_memory_gb - 8:
        return 'ddp'

    # Single GPU fallback
    return 'single'


def recommend_batch_size(model_size: str, use_specboost: bool, gpu_memory_gb: float,
                         n_gpus: int, strategy: str) -> int:
    """Recommend batch size based on strategy and memory."""
    params_m = get_model_params_millions(model_size, use_specboost)

    if strategy == 'fsdp':
        # FSDP shards model, so we can use larger batches
        # Model memory is distributed across GPUs
        model_mem_per_gpu = params_m * 4 / 1000 / n_gpus  # Only weights, sharded
        available = gpu_memory_gb - model_mem_per_gpu - 4  # 4GB overhead
        batch_per_gpu = max(1, int(available / 0.3))  # ~300MB per sample
        return min(32, batch_per_gpu) * n_gpus

    elif strategy == 'ddp':
        # DDP replicates model on each GPU
        model_mem = params_m * 16 / 1000  # weights + grads + optimizer
        available = gpu_memory_gb - model_mem - 4
        batch_per_gpu = max(1, int(available / 0.2))
        return min(16, batch_per_gpu) * n_gpus

    else:  # single
        model_mem = params_m * 20 / 1000
        available = gpu_memory_gb - model_mem - 4
        mem_per_sample = 0.4 if use_specboost else 0.25
        return max(1, min(8, int(available / mem_per_sample)))


# ============== DISTRIBUTED SETUP ==============

def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is main (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# ============== IMPORTS ==============

from src.model.fno_v2 import (
    create_enhanced_model,
    CombinedLossV2,
)
from src.training.dataset import ProppantDataset, create_dataloaders
from src.training.train import Metrics


# ============== TRAINER ==============

class Trainer:
    """
    Unified trainer supporting single GPU, DDP, and FSDP.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 1e-3,
        n_epochs: int = 500,
        checkpoint_dir: Path = None,
        patience: int = 50,
        gradient_accumulation_steps: int = 4,
        warmup_epochs: int = 10,
        strategy: str = 'single',
        rank: int = 0,
        world_size: int = 1,
    ):
        self.device = device
        self.n_epochs = n_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_epochs = warmup_epochs
        self.strategy = strategy
        self.rank = rank
        self.world_size = world_size

        # Wrap model based on strategy
        if strategy == 'fsdp':
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            # Auto-wrap policy: wrap modules with >1M params
            auto_wrap_policy = partial(
                size_based_auto_wrap_policy,
                min_num_params=1_000_000
            )

            self.model = FSDP(
                model.to(device),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=auto_wrap_policy,
                device_id=rank,
                use_orig_params=True,
            )

        elif strategy == 'ddp':
            model = model.to(device)
            self.model = DDP(model, device_ids=[rank])

        else:  # single
            self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        # Scheduler: warmup + cosine
        total_steps = n_epochs * len(train_loader) // gradient_accumulation_steps
        warmup_steps = warmup_epochs * len(train_loader) // gradient_accumulation_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Loss
        self.criterion = CombinedLossV2()

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir and is_main_process():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # History
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_metrics = {}

        # Early stopping
        self.patience = patience
        self.patience_counter = 0

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training", leave=False,
                    disable=not is_main_process())

        for batch_idx, batch in enumerate(pbar):
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)

            pred = self.model(params)
            n_times = pred.shape[1]
            target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)

            loss = self.criterion(pred, target)
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation_steps:.4e}'})

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def validate(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        batch_metrics = []

        for batch in self.val_loader:
            params = batch['params'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)

            pred = self.model(params)
            n_times = pred.shape[1]
            target = trajectory[:, :n_times, :, :].permute(0, 1, 3, 2)

            loss = self.criterion(pred, target)
            total_loss += loss.item()

            metrics = Metrics.compute_all(pred.float(), target.float())
            batch_metrics.append(metrics)

        avg_loss = total_loss / max(1, len(self.val_loader))
        avg_metrics = Metrics.aggregate(batch_metrics)

        return avg_loss, avg_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if not self.checkpoint_dir or not is_main_process():
            return

        # Get state dict (handles FSDP/DDP automatically)
        if self.strategy == 'fsdp':
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                model_state = self.model.state_dict()
        elif hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics,
        }

        # Save with compression
        try:
            torch.save(checkpoint, self.checkpoint_dir / 'last_v2.pt',
                      _use_new_zipfile_serialization=True)
            if is_best:
                torch.save(checkpoint, self.checkpoint_dir / 'best_v2.pt',
                          _use_new_zipfile_serialization=True)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
            # Try saving just the model
            try:
                torch.save(model_state, self.checkpoint_dir / 'model_weights.pt')
                if is_best:
                    torch.save(model_state, self.checkpoint_dir / 'best_weights.pt')
            except:
                pass

    def train(self):
        if is_main_process():
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            n_params = sum(p.numel() for p in base_model.parameters())

            print("=" * 80)
            print("FNO v2 Training")
            print("=" * 80)
            print(f"Strategy: {self.strategy.upper()}")
            print(f"Parameters: {n_params:,}")
            print(f"GPUs: {self.world_size}")
            print(f"Gradient Accumulation: {self.gradient_accumulation_steps}x")
            print(f"Effective Batch: {self.train_loader.batch_size * self.gradient_accumulation_steps}")
            print("-" * 80)

        for epoch in range(1, self.n_epochs + 1):
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            t0 = time.time()

            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            is_best = val_loss < self.best_val_loss - 1e-5
            if is_best:
                self.best_val_loss = val_loss
                self.best_metrics = val_metrics
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            if is_main_process():
                elapsed = time.time() - t0
                lr = self.optimizer.param_groups[0]['lr']
                status = " [BEST]" if is_best else f" [{self.patience_counter}/{self.patience}]"

                print(f"Epoch {epoch:4d}/{self.n_epochs} | "
                      f"Train: {train_loss:.4e} | Val: {val_loss:.4e} | "
                      f"LR: {lr:.2e} | Time: {elapsed:.1f}s{status}")
                print(f"         → {Metrics.format(val_metrics)}")

            if self.patience_counter >= self.patience:
                if is_main_process():
                    print(f"\nEarly stopping at epoch {epoch}")
                break

        if is_main_process():
            print("=" * 80)
            print(f"Training Complete! Best loss: {self.best_val_loss:.4e}")
            print("=" * 80)


# ============== MAIN ==============

def main(
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 0,
    model_size: str = 'large',
    use_specboost: bool = False,
    patience: int = 50,
):
    """Main training entry point."""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'

    # GPU info
    gpu_info = get_gpu_info()

    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)

    if not gpu_info['available']:
        print("No GPU available, using CPU")
        device = 'cpu'
        strategy = 'single'
        n_gpus = 1
        if batch_size <= 0:
            batch_size = 4
    else:
        device = 'cuda'
        n_gpus = gpu_info['count']
        min_gpu_mem = min(g['memory_gb'] for g in gpu_info['devices'])

        for gpu in gpu_info['devices']:
            print(f"  [{gpu['id']}] {gpu['name']} - {gpu['memory_gb']:.1f} GB "
                  f"(free: {gpu['free_memory_gb']:.1f} GB)")
        print(f"Total VRAM: {gpu_info['total_memory_gb']:.1f} GB")

        # Choose strategy
        strategy = recommend_strategy(model_size, use_specboost, n_gpus, min_gpu_mem)
        print(f"\nSelected strategy: {strategy.upper()}")

        # Auto batch size
        if batch_size <= 0:
            batch_size = recommend_batch_size(model_size, use_specboost, min_gpu_mem,
                                             n_gpus, strategy)
        print(f"Batch size: {batch_size}")

    print("=" * 60)

    # Check data
    if not (data_dir / 'metadata.json').exists():
        print(f"No data found in {data_dir}")
        print("Run: python app.py --generate --samples 500")
        return

    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    # Create dataloaders
    num_workers = min(4, n_gpus * 2) if gpu_info['available'] else 0
    train_loader, val_loader = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        train_ratio=0.8,
        num_workers=num_workers
    )

    sample = next(iter(train_loader))
    n_times = sample['trajectory'].shape[1]
    n_params = sample['params'].shape[1]

    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Grid: {metadata['nx']}x{metadata['ny']}, {n_times} time steps")
    print(f"Model: {model_size}" + (" + SpecBoost" if use_specboost else ""))

    # For FSDP/DDP, we need distributed setup
    if strategy in ['fsdp', 'ddp'] and n_gpus > 1:
        # Launch with torchrun or use spawn
        import torch.multiprocessing as mp

        print(f"\nLaunching {strategy.upper()} training on {n_gpus} GPUs...")
        print("Use: torchrun --nproc_per_node=8 -m src.training.train_v2")

        # For simplicity, fall back to single GPU if not launched with torchrun
        if not dist.is_initialized():
            print("Falling back to single GPU (use torchrun for multi-GPU)")
            strategy = 'single'

    # Create model
    model = create_enhanced_model(
        nx=metadata['nx'],
        ny=metadata['ny'],
        n_times=n_times,
        n_params=n_params,
        device='cpu',  # Move to GPU in trainer
        model_size=model_size,
        use_specboost=use_specboost,
    )

    # Enable gradient checkpointing for large models
    if model_size in ['large', 'xlarge'] or use_specboost:
        print("Enabling gradient checkpointing...")
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()

    # Gradient accumulation
    grad_accum = 8 if strategy == 'single' else 4

    # Scale LR
    effective_batch = batch_size * grad_accum
    scaled_lr = lr * (effective_batch / 128)
    print(f"Effective batch: {effective_batch}, LR: {scaled_lr:.6f}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=scaled_lr,
        n_epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        gradient_accumulation_steps=grad_accum,
        strategy=strategy,
        rank=0,
        world_size=1,
    )

    trainer.train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train FNO v2')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=0)
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
