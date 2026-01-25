"""
Dataset generation and loading for FNO training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional
import json
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.solver.advection_diffusion import (
    SimulationParams,
    AdvectionDiffusionSolver,
    create_initial_condition,
    generate_training_sample
)


class ProppantDataset(Dataset):
    """Dataset of advection-diffusion simulations."""

    def __init__(
        self,
        data_dir: Path,
        transform=None
    ):
        """
        Args:
            data_dir: Directory containing .npz files
            transform: Optional transform to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Find all data files
        self.files = sorted(self.data_dir.glob("sample_*.npz"))

        if len(self.files) == 0:
            raise ValueError(f"No data files found in {data_dir}")

        # Load first file to get dimensions
        with np.load(self.files[0]) as data:
            self.n_times = data['concentrations'].shape[0]
            self.nx = data['concentrations'].shape[1]
            self.ny = data['concentrations'].shape[2]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        with np.load(self.files[idx]) as data:
            c0 = data['concentrations'][0]  # Initial condition
            trajectory = data['concentrations']  # Full trajectory
            u = data['velocity_u']
            v = data['velocity_v']

        # Convert to tensors
        c0 = torch.from_numpy(c0).float()
        trajectory = torch.from_numpy(trajectory).float()
        u = torch.from_numpy(u).float()
        v = torch.from_numpy(v).float()

        sample = {
            'c0': c0,
            'trajectory': trajectory,
            'u': u,
            'v': v
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def generate_dataset(
    output_dir: Path,
    n_samples: int = 1000,
    params: Optional[SimulationParams] = None,
    seed: int = 42
):
    """
    Generate synthetic training dataset.

    Args:
        output_dir: Directory to save data
        n_samples: Number of samples to generate
        params: Simulation parameters (uses defaults if None)
        seed: Random seed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if params is None:
        params = SimulationParams(
            nx=128, ny=64,
            T=4.0, dt=0.001,
            D=0.01, U_max=1.0,
            save_every=100  # ~40 frames for T=4.0
        )

    np.random.seed(seed)

    # Save metadata
    metadata = {
        'n_samples': n_samples,
        'nx': params.nx,
        'ny': params.ny,
        'T': params.T,
        'dt': params.dt,
        'D': params.D,
        'U_max': params.U_max,
        'save_every': params.save_every,
        'Lx': params.Lx,
        'Ly': params.Ly
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    for i in tqdm(range(n_samples), desc="Generating samples"):
        # Randomize initial conditions
        ic_type = np.random.choice(['gaussian', 'step', 'random'], p=[0.6, 0.2, 0.2])

        if ic_type == 'gaussian':
            ic_params = {
                'ic_type': 'gaussian',
                'x0': np.random.uniform(0.1, 0.5),
                'y0': np.random.uniform(0.2, 0.8),
                'sigma_x': np.random.uniform(0.05, 0.2),
                'sigma_y': np.random.uniform(0.05, 0.2),
                'amplitude': np.random.uniform(0.3, 0.8)
            }
        elif ic_type == 'step':
            ic_params = {
                'ic_type': 'step',
                'x_thresh': np.random.uniform(0.2, 0.6),
                'amplitude': np.random.uniform(0.3, 0.7)
            }
        else:
            ic_params = {
                'ic_type': 'random',
                'amplitude': np.random.uniform(0.3, 0.6),
                'seed': i
            }

        # Optionally vary physics parameters
        sample_params = SimulationParams(
            nx=params.nx, ny=params.ny,
            Lx=params.Lx, Ly=params.Ly,
            T=params.T, dt=params.dt,
            D=np.random.uniform(0.005, 0.02),  # Vary diffusion
            U_max=np.random.uniform(0.5, 1.5),  # Vary velocity
            save_every=params.save_every
        )

        sample = generate_training_sample(sample_params, ic_params)

        # Save
        np.savez_compressed(
            output_dir / f'sample_{i:05d}.npz',
            concentrations=sample['concentrations'].astype(np.float32),
            velocity_u=sample['velocity_u'].astype(np.float32),
            velocity_v=sample['velocity_v'].astype(np.float32),
            times=sample['times'].astype(np.float32)
        )

    print(f"Generated {n_samples} samples in {output_dir}")


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 16,
    train_ratio: float = 0.8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    """
    dataset = ProppantDataset(data_dir)

    # Split
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
    # Generate small test dataset
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    generate_dataset(output_dir, n_samples=10)

    # Test loading
    dataset = ProppantDataset(output_dir)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"c0 shape: {sample['c0'].shape}")
    print(f"trajectory shape: {sample['trajectory'].shape}")
