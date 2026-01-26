"""
Dataset generation and loading for FNO training.
Uses the full CPU_solver with proper physics.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.solver.solver_wrapper import ProppantSolver


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
            self.ny = data['concentrations'].shape[1]
            self.nx = data['concentrations'].shape[2]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        with np.load(self.files[idx]) as data:
            trajectory = data['concentrations']
            params = data['params']

        trajectory = torch.from_numpy(trajectory).float()
        params = torch.from_numpy(params).float()

        sample = {
            'trajectory': trajectory,
            'params': params,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def generate_training_sample(
    c_inlet: float,
    Q_inlet: float,
    g: float,
    mu0: float,
    r_particle: float,
    nx: int = 50,
    ny: int = 25,
    T: float = 80.0,
    dT: float = 4.0,
) -> dict:
    """Generate a single training sample."""

    solver = ProppantSolver(
        nx=nx, ny=ny,
        Lx=60.0, Ly=30.0,
        T=T, dT=dT,
        c_inlet=c_inlet,
        Q_inlet=Q_inlet,
        g=g,
        mu0=mu0,
        r_particle=r_particle,
    )

    times, concentrations = solver.solve()

    # Clip to valid range
    concentrations = np.clip(concentrations, 0, 0.635)

    return {
        'concentrations': concentrations.astype(np.float32),
        'times': times.astype(np.float32),
        'params': np.array([c_inlet, Q_inlet, g, mu0, r_particle], dtype=np.float32)
    }


def generate_dataset(
    output_dir: Path,
    n_samples: int = 100,
    seed: int = 42
):
    """
    Generate training dataset with varied physical parameters.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    # Grid parameters
    nx, ny = 50, 25
    T, dT = 80.0, 4.0

    metadata = {
        'n_samples': n_samples,
        'nx': nx,
        'ny': ny,
        'Lx': 60.0,
        'Ly': 30.0,
        'T': T,
        'dT': dT,
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    for i in tqdm(range(n_samples), desc="Generating samples"):
        # Randomize physical parameters
        c_inlet = np.random.uniform(0.2, 0.5)
        Q_inlet = np.random.uniform(0.03, 0.08)
        g = np.random.uniform(0, 12)
        mu0 = np.random.uniform(0.0005, 0.005)
        r_particle = np.random.uniform(0.0001, 0.0004)

        try:
            sample = generate_training_sample(
                c_inlet=c_inlet,
                Q_inlet=Q_inlet,
                g=g,
                mu0=mu0,
                r_particle=r_particle,
                nx=nx, ny=ny, T=T, dT=dT
            )

            np.savez_compressed(
                output_dir / f'sample_{i:05d}.npz',
                concentrations=sample['concentrations'],
                times=sample['times'],
                params=sample['params']
            )
        except Exception as e:
            print(f"Sample {i} failed: {e}")
            continue

    print(f"Generated {n_samples} samples in {output_dir}")


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 8,
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
    generate_dataset(output_dir, n_samples=5)

    dataset = ProppantDataset(output_dir)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Trajectory shape: {sample['trajectory'].shape}")
    print(f"Params: {sample['params']}")
