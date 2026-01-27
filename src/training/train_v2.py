"""
Enhanced training script for FNO v2 with automatic multi-GPU support.
Uses torch.multiprocessing.spawn for distributed training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os
import math
from pathlib import Path
import json
import time
from tqdm import tqdm
from functools import partial

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


# ============== GPU UTILITIES ==============

def get_available_gpus(min_free_gb: float = 10.0):
    """
    Get list of GPU indices with sufficient free memory.

    Args:
        min_free_gb: Minimum free memory in GB required

    Returns:
        List of available GPU indices
    """
    if not torch.cuda.is_available():
        return []

    available = []
    count = torch.cuda.device_count()

    for i in range(count):
        try:
            # Get memory info
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            free_gb = free_mem / (1024**3)

            if free_gb >= min_free_gb:
                available.append(i)
        except Exception as e:
            print(f"  Warning: Could not check GPU {i}: {e}")
            continue

    return available


def get_gpu_count():
    """Get number of available GPUs with sufficient memory."""
    return len(get_available_gpus())


def print_gpu_info(min_free_gb: float = 10.0):
    """Print GPU information and availability."""
    if not torch.cuda.is_available():
        print("No GPU available")
        return []

    count = torch.cuda.device_count()
    available = []

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)

        try:
            free_mem, _ = torch.cuda.mem_get_info(i)
            free_gb = free_mem / (1024**3)
            status = "OK" if free_gb >= min_free_gb else f"BUSY ({total_gb - free_gb:.1f} GB used)"

            if free_gb >= min_free_gb:
                available.append(i)
                print(f"  [{i}] {props.name} - {free_gb:.1f}/{total_gb:.1f} GB free [OK]")
            else:
                print(f"  [{i}] {props.name} - {free_gb:.1f}/{total_gb:.1f} GB free [SKIP]")
        except Exception as e:
            print(f"  [{i}] {props.name} - {total_gb:.1f} GB [ERROR: {e}]")

    print(f"Available: {len(available)}/{count} GPUs")
    return available


# ============== DISTRIBUTED SETUP ==============

def setup_distributed(rank, world_size, gpu_ids):
    """Initialize distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Map rank to actual GPU ID
    gpu_id = gpu_ids[rank]
    torch.cuda.set_device(gpu_id)
    return gpu_id


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============== IMPORTS ==============

from src.model.fno_v2 import create_enhanced_model, CombinedLossV2
from src.training.dataset import ProppantDataset, create_dataloaders
from src.training.train import Metrics


# ============== DISTRIBUTED TRAINER ==============

def train_worker(rank, world_size, gpu_ids, args):
    """
    Training worker for distributed training.
    Each GPU runs this function.
    """
    # Setup distributed with GPU mapping
    gpu_id = setup_distributed(rank, world_size, gpu_ids)

    device = f'cuda:{gpu_id}'
    is_main = (rank == 0)

    if is_main:
        print(f"\n{'='*60}")
        print(f"Starting distributed training on {world_size} GPUs")
        print(f"{'='*60}")

    # Load data
    data_dir = Path(args['data_dir'])
    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    # Create dataset
    dataset = ProppantDataset(data_dir)

    # Split into train/val
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # DataLoaders
    batch_size_per_gpu = args['batch_size'] // world_size
    batch_size_per_gpu = max(1, batch_size_per_gpu)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=0,  # Avoid /tmp issues on shared systems
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    if is_main:
        sample = dataset[0]
        n_times = sample['trajectory'].shape[0]
        n_params = sample['params'].shape[0]
        print(f"Data: {n_train} train, {n_val} val")
        print(f"Batch per GPU: {batch_size_per_gpu}")
        print(f"Grid: {metadata['nx']}x{metadata['ny']}, {n_times} time steps")

    # Create model
    sample = dataset[0]
    n_times = sample['trajectory'].shape[0]
    n_params = sample['params'].shape[0]

    model = create_enhanced_model(
        nx=metadata['nx'],
        ny=metadata['ny'],
        n_times=n_times,
        n_params=n_params,
        device='cpu',
        model_size=args['model_size'],
        use_specboost=args['use_specboost'],
    )

    # Move to GPU and wrap with DDP
    # find_unused_parameters=True needed for SpecBoost (residual stages)
    model = model.to(device)
    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    n_params_total = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model parameters: {n_params_total:,}")

    # Optimizer and scheduler
    lr = args['lr']
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_epochs = args['epochs']
    total_steps = n_epochs * len(train_loader)
    warmup_steps = min(10, n_epochs // 10) * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    criterion = CombinedLossV2()

    # Checkpoint dir
    checkpoint_dir = Path(args['checkpoint_dir'])
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = args['patience']

    if is_main:
        print(f"\n{'='*60}")
        print("Training started")
        print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()

        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main, leave=False)

        for batch in pbar:
            params = batch['params'].to(device)
            trajectory = batch['trajectory'].to(device)

            optimizer.zero_grad()

            pred = model(params)
            n_t = pred.shape[1]
            target = trajectory[:, :n_t, :, :].permute(0, 1, 3, 2)

            loss = criterion(pred, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4e}'})

        train_loss = total_loss / max(1, n_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        batch_metrics = []

        with torch.no_grad():
            for batch in val_loader:
                params = batch['params'].to(device)
                trajectory = batch['trajectory'].to(device)

                pred = model(params)
                n_t = pred.shape[1]
                target = trajectory[:, :n_t, :, :].permute(0, 1, 3, 2)

                loss = criterion(pred, target)
                val_loss += loss.item()
                val_batches += 1

                metrics = Metrics.compute_all(pred.float(), target.float())
                batch_metrics.append(metrics)

        val_loss = val_loss / max(1, val_batches)
        avg_metrics = Metrics.aggregate(batch_metrics)

        # Gather val_loss from all ranks
        val_loss_tensor = torch.tensor([val_loss], device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_loss_tensor.item()

        # Check best
        is_best = val_loss < best_val_loss - 1e-5
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint (only rank 0)
        if is_main:
            elapsed = time.time() - t0
            lr_current = optimizer.param_groups[0]['lr']
            status = " [BEST]" if is_best else f" [{patience_counter}/{patience}]"

            print(f"Epoch {epoch:4d}/{n_epochs} | "
                  f"Train: {train_loss:.4e} | Val: {val_loss:.4e} | "
                  f"LR: {lr_current:.2e} | Time: {elapsed:.1f}s{status}")
            print(f"         → {Metrics.format(avg_metrics)}")

            # Save checkpoint
            try:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, checkpoint_dir / 'last_v2.pt')
                if is_best:
                    torch.save(checkpoint, checkpoint_dir / 'best_v2.pt')
            except Exception as e:
                print(f"Warning: Could not save checkpoint: {e}")

        # Early stopping
        if patience_counter >= patience:
            if is_main:
                print(f"\nEarly stopping at epoch {epoch}")
            break

        # Sync all processes
        dist.barrier()

    if is_main:
        print(f"\n{'='*60}")
        print(f"Training complete! Best val loss: {best_val_loss:.4e}")
        print(f"{'='*60}")

    cleanup_distributed()


# ============== MAIN ==============

def main(
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 0,
    model_size: str = 'large',
    use_specboost: bool = False,
    patience: int = 50,
):
    """Main entry point - launches distributed training."""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'

    # Check data
    if not (data_dir / 'metadata.json').exists():
        print(f"No data found in {data_dir}")
        print("Run: python app.py --generate --samples 500")
        return

    # Minimum free memory required per GPU (in GB)
    # xlarge+specboost needs ~25GB, large needs ~10GB
    if model_size == 'xlarge' and use_specboost:
        min_free_gb = 25.0
    elif model_size == 'xlarge':
        min_free_gb = 20.0
    elif model_size == 'large':
        min_free_gb = 12.0
    else:
        min_free_gb = 8.0

    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)

    # Get available GPUs with sufficient memory
    gpu_ids = print_gpu_info(min_free_gb=min_free_gb)
    n_gpus = len(gpu_ids)

    if n_gpus == 0:
        print(f"\nNo GPUs with >= {min_free_gb:.0f} GB free memory!")
        print("Try a smaller model or free up GPU memory.")
        return

    print(f"\nUsing GPUs: {gpu_ids}")

    # Auto batch size
    if batch_size <= 0:
        # Conservative: 4 samples per GPU for xlarge+specboost
        if model_size == 'xlarge' and use_specboost:
            batch_size = 4 * n_gpus
        elif model_size == 'xlarge' or use_specboost:
            batch_size = 8 * n_gpus
        elif model_size == 'large':
            batch_size = 16 * n_gpus
        else:
            batch_size = 32 * n_gpus

    print(f"Model: {model_size}" + (" + SpecBoost" if use_specboost else ""))
    print(f"GPUs: {n_gpus} (IDs: {gpu_ids})")
    print(f"Total batch size: {batch_size}")
    print(f"Batch per GPU: {batch_size // n_gpus}")

    # Training args
    args = {
        'data_dir': str(data_dir),
        'checkpoint_dir': str(checkpoint_dir),
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'model_size': model_size,
        'use_specboost': use_specboost,
        'patience': patience,
    }

    # Launch distributed training
    print(f"\nLaunching {n_gpus}-GPU distributed training...")
    mp.spawn(
        train_worker,
        args=(n_gpus, gpu_ids, args),
        nprocs=n_gpus,
        join=True
    )


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
