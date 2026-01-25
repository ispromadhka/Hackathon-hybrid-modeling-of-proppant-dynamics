"""
Dataset generation and loading for FNO training.
Uses the proppant transport solver with inlet injection.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional
import json
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.solver.proppant_transport import (
    SimulationParams,
    PhysicalParams,
    ProppantTransportSolver,
)


class ProppantDataset(Dataset):
    """Dataset of proppant transport simulations."""

    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        self.files = sorted(self.data_dir.glob("sample_*.npz"))

        if len(self.files) == 0:
            raise ValueError(f"No data files found in {data_dir}")

        with np.load(self.files[0]) as data:
            self.n_times = data['concentrations'].shape[0]
            self.nx = data['concentrations'].shape[1]
            self.ny = data['concentrations'].shape[2]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        with np.load(self.files[idx]) as data:
            c0 = data['concentrations'][0]
            trajectory = data['concentrations']
            params = data['params']  # [c_inlet, U_max, g, mu_f, d_p]

        c0 = torch.from_numpy(c0).float()
        trajectory = torch.from_numpy(trajectory).float()
        params = torch.from_numpy(params).float()

        sample = {
            'c0': c0,
            'trajectory': trajectory,
            'params': params,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def generate_training_sample(sim_params: SimulationParams, phys_params: PhysicalParams) -> dict:
    """Generate a single training sample."""
    solver = ProppantTransportSolver(sim_params, phys_params)
    times, concentrations = solver.solve()

    return {
        'concentrations': concentrations,
        'times': times,
        'params': np.array([
            sim_params.c_inlet,
            phys_params.U_max,
            phys_params.g,
            phys_params.mu_f,
            phys_params.d_p
        ], dtype=np.float32)
    }


def generate_dataset(
    output_dir: Path,
    n_samples: int = 500,
    seed: int = 42
):
    """
    Generate training dataset with varied physical parameters.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    # Fixed simulation grid
    nx, ny = 64, 32
    T, dt = 2.0, 0.004
    save_every = 20

    metadata = {
        'n_samples': n_samples,
        'nx': nx,
        'ny': ny,
        'Lx': 2.0,
        'Ly': 1.0,
        'T': T,
        'dt': dt,
        'save_every': save_every,
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    for i in tqdm(range(n_samples), desc="Generating samples"):
        # Randomize physical parameters
        c_inlet = np.random.uniform(0.15, 0.45)
        U_max = np.random.uniform(0.4, 1.2)
        g = np.random.uniform(5.0, 15.0)
        mu_f = np.random.uniform(0.005, 0.05)  # 5-50 mPa·s
        d_p = np.random.uniform(0.0002, 0.0008)  # 200-800 μm

        sim_params = SimulationParams(
            nx=nx, ny=ny,
            T=T, dt=dt,
            save_every=save_every,
            c_inlet=c_inlet
        )

        phys_params = PhysicalParams(
            g=g,
            mu_f=mu_f,
            U_max=U_max,
            d_p=d_p
        )

        sample = generate_training_sample(sim_params, phys_params)

        np.savez_compressed(
            output_dir / f'sample_{i:05d}.npz',
            concentrations=sample['concentrations'].astype(np.float32),
            times=sample['times'].astype(np.float32),
            params=sample['params']
        )

    print(f"Generated {n_samples} samples in {output_dir}")


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 16,
    train_ratio: float = 0.8,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    dataset = ProppantDataset(data_dir)

    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    generate_dataset(output_dir, n_samples=10)

    dataset = ProppantDataset(output_dir)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"c0 shape: {sample['c0'].shape}")
    print(f"trajectory shape: {sample['trajectory'].shape}")
    print(f"params: {sample['params']}")
