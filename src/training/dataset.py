"""
Dataset generation and loading for FNO training.
Uses the full CPU_solver with proper physics.

Features:
- Latin Hypercube Sampling for uniform parameter coverage
- Multiple injection modes (continuous, single_pulse, multi_pulse)
- Varied numerical schemes (limiter types, RK stages)
- Parallel generation with multiprocessing
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def latin_hypercube_sampling(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """
    Generate Latin Hypercube samples in [0, 1]^n_dims.
    Provides better coverage of parameter space than random sampling.
    """
    np.random.seed(seed)
    samples = np.zeros((n_samples, n_dims))

    for dim in range(n_dims):
        # Divide [0, 1] into n_samples equal intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        # Sample uniformly within each interval
        for i in range(n_samples):
            samples[i, dim] = np.random.uniform(intervals[i], intervals[i + 1])
        # Shuffle to break correlation between dimensions
        np.random.shuffle(samples[:, dim])

    return samples


def create_injection_pattern(
    mode: str,
    c_inlet: float,
    T: float,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create injection pattern based on mode.

    Modes:
    - 'continuous': constant injection throughout simulation
    - 'single_pulse': inject for first portion, then stop
    - 'multi_pulse': alternating injection on/off
    """
    if seed is not None:
        np.random.seed(seed)

    if mode == 'continuous':
        c_in_times = np.array([T * 100])  # Effectively infinite
        c_in_arr = np.array([c_inlet, 0.0])

    elif mode == 'single_pulse':
        # Inject for 20-60% of simulation time
        pulse_fraction = np.random.uniform(0.2, 0.6)
        pulse_end = T * pulse_fraction
        c_in_times = np.array([pulse_end])
        c_in_arr = np.array([c_inlet, 0.0])

    elif mode == 'multi_pulse':
        # 2-4 pulses
        n_pulses = np.random.randint(2, 5)
        pulse_duration = T / (2 * n_pulses)

        c_in_times = []
        c_in_arr = []

        for i in range(n_pulses):
            # On phase
            c_in_times.append((2 * i + 1) * pulse_duration)
            c_in_arr.append(c_inlet)
            # Off phase (except last)
            if i < n_pulses - 1:
                c_in_times.append((2 * i + 2) * pulse_duration)
                c_in_arr.append(0.0)

        c_in_arr.append(0.0)  # Final state
        c_in_times = np.array(c_in_times)
        c_in_arr = np.array(c_in_arr)
    else:
        raise ValueError(f"Unknown injection mode: {mode}")

    return c_in_times, c_in_arr


def generate_single_sample(args: dict) -> Optional[dict]:
    """Generate a single training sample. Used for parallel processing."""
    try:
        # Extract parameters
        c_inlet = args['c_inlet']
        Q_inlet = args['Q_inlet']
        g = args['g']
        mu0 = args['mu0']
        r_particle = args['r_particle']
        inlet_fraction = args['inlet_fraction']
        rk_stages = args['rk_stages']
        lim_type = args['lim_type']
        injection_mode = args['injection_mode']
        nx = args['nx']
        ny = args['ny']
        T = args['T']
        dT = args['dT']
        sample_idx = args['sample_idx']

        # Create injection pattern
        c_in_times, c_in_arr = create_injection_pattern(
            injection_mode, c_inlet, T, seed=sample_idx
        )

        # Create solver
        solver = ProppantSolver(
            nx=nx, ny=ny,
            Lx=60.0, Ly=30.0,
            T=T, dT=dT,
            c_inlet=c_inlet,
            Q_inlet=Q_inlet,
            g=g,
            mu0=mu0,
            r_particle=r_particle,
            inlet_fraction=inlet_fraction,
            c_in_times=c_in_times,
            c_in_arr=c_in_arr,
            rk_stages=rk_stages,
            lim_type=lim_type,
        )

        # Solve
        times, concentrations = solver.solve()

        # Clip to valid range
        concentrations = np.clip(concentrations, 0, 0.635)

        # Encode categorical parameters
        injection_mode_map = {'continuous': 0, 'single_pulse': 1, 'multi_pulse': 2}
        lim_type_map = {'koren': 0, 'superbee': 1, 'minmod': 2, 'vanleer': 3}

        # Parameters vector for FNO input
        # [c_inlet, Q_inlet, g, mu0, r_particle, inlet_fraction, rk_stages, lim_type_enc, injection_mode_enc]
        params = np.array([
            c_inlet,
            Q_inlet,
            g,
            mu0,
            r_particle,
            inlet_fraction,
            rk_stages,
            lim_type_map.get(lim_type, 0),
            injection_mode_map.get(injection_mode, 0),
        ], dtype=np.float32)

        return {
            'concentrations': concentrations.astype(np.float32),
            'times': times.astype(np.float32),
            'params': params,
            'sample_idx': sample_idx,
            'config': {
                'c_inlet': c_inlet,
                'Q_inlet': Q_inlet,
                'g': g,
                'mu0': mu0,
                'r_particle': r_particle,
                'inlet_fraction': inlet_fraction,
                'rk_stages': rk_stages,
                'lim_type': lim_type,
                'injection_mode': injection_mode,
            }
        }

    except Exception as e:
        print(f"Sample {args.get('sample_idx', '?')} failed: {e}")
        return None


def generate_dataset(
    output_dir: Path,
    n_samples: int = 100,
    seed: int = 42,
    n_workers: int = 1,
    grid_size: Tuple[int, int] = (60, 30),
    T: float = 100.0,
    dT: float = 5.0,
):
    """
    Generate training dataset with varied physical parameters.

    Uses Latin Hypercube Sampling for continuous parameters and
    uniform sampling for categorical parameters.

    Parameters
    ----------
    output_dir : Path
        Output directory for generated samples
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility
    n_workers : int
        Number of parallel workers (1 = sequential)
    grid_size : tuple
        (nx, ny) grid resolution
    T : float
        Total simulation time [s]
    dT : float
        Output time step [s]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    nx, ny = grid_size

    # Parameter ranges (physically realistic)
    param_ranges = {
        'c_inlet': (0.15, 0.50),        # Proppant concentration
        'Q_inlet': (0.02, 0.10),        # Flow rate [m²/s]
        'g': (0.0, 12.0),               # Gravity [m/s²] (0 = horizontal, 9.81 = vertical)
        'mu0': (0.0005, 0.010),         # Viscosity [Pa·s] (0.5-10 mPa·s)
        'r_particle': (0.0001, 0.0005), # Particle radius [m] (100-500 μm)
        'inlet_fraction': (1/6, 1/2),   # Inlet height fraction
    }

    # Categorical parameters
    injection_modes = ['continuous', 'single_pulse', 'multi_pulse']
    lim_types = ['koren', 'superbee', 'minmod']
    rk_stages_options = [2, 3]

    # Generate Latin Hypercube samples for continuous parameters
    n_continuous = len(param_ranges)
    lhs_samples = latin_hypercube_sampling(n_samples, n_continuous, seed)

    # Scale to parameter ranges
    param_names = list(param_ranges.keys())
    scaled_params = np.zeros((n_samples, n_continuous))
    for i, name in enumerate(param_names):
        low, high = param_ranges[name]
        scaled_params[:, i] = low + lhs_samples[:, i] * (high - low)

    # Prepare all sample configurations
    sample_configs = []
    for i in range(n_samples):
        config = {
            'c_inlet': scaled_params[i, 0],
            'Q_inlet': scaled_params[i, 1],
            'g': scaled_params[i, 2],
            'mu0': scaled_params[i, 3],
            'r_particle': scaled_params[i, 4],
            'inlet_fraction': scaled_params[i, 5],
            'rk_stages': np.random.choice(rk_stages_options),
            'lim_type': np.random.choice(lim_types),
            'injection_mode': np.random.choice(injection_modes),
            'nx': nx,
            'ny': ny,
            'T': T,
            'dT': dT,
            'sample_idx': i,
        }
        sample_configs.append(config)

    # Save metadata
    metadata = {
        'n_samples': n_samples,
        'nx': nx,
        'ny': ny,
        'Lx': 60.0,
        'Ly': 30.0,
        'T': T,
        'dT': dT,
        'param_ranges': param_ranges,
        'injection_modes': injection_modes,
        'lim_types': lim_types,
        'rk_stages_options': rk_stages_options,
        'seed': seed,
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Generate samples
    successful = 0
    failed = 0

    if n_workers > 1:
        # Parallel generation
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(generate_single_sample, cfg): cfg['sample_idx']
                      for cfg in sample_configs}

            for future in tqdm(as_completed(futures), total=n_samples, desc="Generating samples"):
                result = future.result()
                if result is not None:
                    idx = result['sample_idx']
                    np.savez_compressed(
                        output_dir / f'sample_{idx:05d}.npz',
                        concentrations=result['concentrations'],
                        times=result['times'],
                        params=result['params'],
                    )
                    successful += 1
                else:
                    failed += 1
    else:
        # Sequential generation
        for config in tqdm(sample_configs, desc="Generating samples"):
            result = generate_single_sample(config)
            if result is not None:
                idx = result['sample_idx']
                np.savez_compressed(
                    output_dir / f'sample_{idx:05d}.npz',
                    concentrations=result['concentrations'],
                    times=result['times'],
                    params=result['params'],
                )
                successful += 1
            else:
                failed += 1

    print(f"Generated {successful} samples ({failed} failed) in {output_dir}")
    return successful


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

    # Test with small dataset
    generate_dataset(
        output_dir,
        n_samples=5,
        grid_size=(50, 25),
        T=80.0,
        dT=4.0,
    )

    dataset = ProppantDataset(output_dir)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Trajectory shape: {sample['trajectory'].shape}")
    print(f"Params shape: {sample['params'].shape}")
    print(f"Params: {sample['params']}")
