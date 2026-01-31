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
import math
import itertools

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.solver.solver_wrapper import ProppantSolver
from src.solver.generation import generate_simulations
from src.solver.generation import load_generation_config, _values_from_spec, generate_for_params
from src.solver.to_torch import build_torch_data


def validate_params(params_raw: np.ndarray, gen_cfg: dict) -> bool:
    """Проверка физической корректности параметров."""
    if len(params_raw) < 7:
        return False
    c_in, w0, mu0, Q, chi, c_in_times, dT = params_raw[:7]

    cmax = float(gen_cfg.get('physics', {}).get('cmax', 0.635))

    if w0 <= 0 or mu0 <= 0 or chi <= 0 or c_in_times <= 0 or dT <= 0:
        return False
    if c_in < 0 or c_in > cmax:
        return False
    if Q >= 0:
        return False
    return True


class ProppantDataset(Dataset):
    """Dataset of proppant transport simulations."""

    def __init__(self, data_dir: Path, transform=None, gen_cfg: dict = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.gen_cfg = gen_cfg

        self.files = sorted(self.data_dir.glob("sample_*.npz"))

        if len(self.files) == 0:
            raise ValueError(f"No data files found in {data_dir}")

        with np.load(self.files[0]) as data:
            self.n_times = data['concentrations'].shape[0]
            self.ny = data['concentrations'].shape[1]
            self.nx = data['concentrations'].shape[2]

        if self.gen_cfg:
            valid_files = []
            invalid_files = []
            for f in self.files:
                try:
                    with np.load(f) as data:
                        if 'params_raw' in data:
                            if validate_params(data['params_raw'], self.gen_cfg):
                                valid_files.append(f)
                            else:
                                invalid_files.append(f)
                        else:
                            valid_files.append(f)
                except Exception:
                    invalid_files.append(f)

            if invalid_files:
                print(f"Warning: Skipping {len(invalid_files)} files with invalid parameters:")
                for f in invalid_files[:5]:
                    print(f"  - {f.name}")
                if len(invalid_files) > 5:
                    print(f"  ... and {len(invalid_files) - 5} more")

            self.files = valid_files
            if len(self.files) == 0:
                raise ValueError(f"No valid data files found in {data_dir}")

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

        # Parameters vector for FNO input - NORMALIZED to [0, 1]
        # This is critical for training stability
        params = np.array([
            c_inlet / 0.5,                    # c_inlet: [0.15, 0.50] -> [0.3, 1.0]
            Q_inlet / 0.1,                    # Q_inlet: [0.02, 0.10] -> [0.2, 1.0]
            g / 12.0,                         # g: [0, 12] -> [0, 1]
            mu0 / 0.01,                       # mu0: [0.0005, 0.01] -> [0.05, 1.0]
            r_particle / 0.0005,              # r_particle: [0.0001, 0.0005] -> [0.2, 1.0]
            inlet_fraction * 2,               # inlet_fraction: [1/6, 1/2] -> [0.33, 1.0]
            (rk_stages - 2) / 1.0,            # rk_stages: [2, 3] -> [0, 1]
            lim_type_map.get(lim_type, 0) / 2.0,  # lim_type: [0, 2] -> [0, 1]
            injection_mode_map.get(injection_mode, 0) / 2.0,  # injection_mode: [0, 2] -> [0, 1]
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
    config_path: Path | None = None,
    clear_processed: bool = False,
):
    output_dir = Path(output_dir)
    project_root = output_dir.parent.parent
    gen_cfg = load_generation_config(config_path=config_path, project_root=project_root)
    cfg = {}
    if config_path is None:
        config_path = project_root / 'configs' / 'default.json'
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    ds_cfg = cfg.get('dataset_generation', {})
    sampling = str(ds_cfg.get('sampling', '')).lower()
    ds_seed = int(ds_cfg.get('seed', seed))

    if sampling == 'random' or sampling == 'lhs':
        p = gen_cfg.get('params', {})
        keys = ['c_in', 'w0', 'mu0', 'Q', 'chi', 'c_in_times']
        grids = []
        for k in keys:
            vals = _values_from_spec(p.get(k))
            if len(vals) == 0:
                raise ValueError(f"Empty parameter grid for {k}")
            grids.append(vals)
        dT_vals = p.get('dT_values', [])
        if len(dT_vals) == 0:
            raise ValueError("Empty dT_values")

        total_combos = int(np.prod([len(v) for v in grids]) * max(1, len(dT_vals)))
        target_n = int(n_samples)
        if target_n > total_combos:
            raise ValueError(f"Requested n_samples={target_n} exceeds total discrete combinations={total_combos}")

        def _params_to_path(params: tuple, root: Path) -> Path:
            c_in, w0, mu0, Q, chi, c_in_times, dT = params
            return root / 'simulation_timeseries' / f"c{c_in:.3f}_w{w0:.3f}_mu{mu0:.3f}_Q{Q:.3f}_chi{chi:.1f}_t{c_in_times:.0f}_dT{dT:.1f}_series.npz"

        def _get_existing_params(timeseries_root: Path) -> set:
            """Получить множество существующих комбинаций параметров."""
            existing = set()
            for npz_file in timeseries_root.glob("*_series.npz"):
                try:
                    name = npz_file.stem.replace("_series", "")
                    parts = name.split("_")
                    if len(parts) >= 7:
                        c_in = float(parts[0][1:])
                        w0 = float(parts[1][1:])
                        mu0 = float(parts[2][2:])
                        Q = float(parts[3][1:])
                        chi = float(parts[4][3:])
                        c_in_times = float(parts[5][1:])
                        dT = float(parts[6][2:])
                        existing.add((c_in, w0, mu0, Q, chi, c_in_times, dT))
                except Exception:
                    continue
            return existing

        params_set: set[tuple] = set()
        timeseries_root = project_root / 'simulation_timeseries'
        timeseries_root.mkdir(parents=True, exist_ok=True)
        existing_params = _get_existing_params(timeseries_root)

        rng = np.random.default_rng(ds_seed)
        attempt = 0
        max_attempts = target_n * 100

        while len(params_set) < target_n and attempt < max_attempts:
            if sampling == 'random':
                chosen = []
                for j in range(len(keys)):
                    vals = grids[j]
                    idx = rng.integers(0, len(vals))
                    chosen.append(float(vals[idx]))
                dT_idx = rng.integers(0, len(dT_vals))
                chosen.append(float(dT_vals[dT_idx]))
            else:
                batch = max(256, target_n * 4)
                u = latin_hypercube_sampling(batch, len(keys), seed=ds_seed + attempt)
                i = attempt % batch
                chosen = []
                for j in range(len(keys)):
                    vals = grids[j]
                    idx = int(math.floor(float(u[i, j]) * len(vals)))
                    if idx >= len(vals):
                        idx = len(vals) - 1
                    chosen.append(float(vals[idx]))
                dT = float(dT_vals[(len(params_set) + attempt) % len(dT_vals)])
                chosen.append(dT)

            t = tuple(chosen)
            if not validate_params(np.array(t), gen_cfg):
                attempt += 1
                continue
            if t not in existing_params and t not in params_set:
                if _params_to_path(t, timeseries_root).exists():
                    existing_params.add(t)
                    continue
                params_set.add(t)
            attempt += 1

        if len(params_set) < target_n:
            for combo in itertools.product(*[list(map(float, v)) for v in grids], [float(v) for v in dT_vals]):
                t = tuple(combo)
                if not validate_params(np.array(t), gen_cfg):
                    continue
                if t in existing_params or t in params_set:
                    continue
                if _params_to_path(t, timeseries_root).exists():
                    existing_params.add(t)
                    continue
                params_set.add(t)
                if len(params_set) >= target_n:
                    break

        if len(params_set) < target_n:
            raise RuntimeError(f"Not enough new combinations to add {target_n} samples (available new={len(params_set)})")
        params_list = list(params_set)[:target_n]

        pipeline = tqdm(total=3, desc="Dataset pipeline")
        try:
            pipeline.set_postfix_str("generate")
            expected_paths = [_params_to_path(p, timeseries_root) for p in params_list]
            missing = [p for p in params_list if not _params_to_path(p, timeseries_root).exists()]
            tries = 0
            max_tries = 10
            while missing and tries < max_tries:
                if tries > 0:
                    print(f"\nRetry {tries}/{max_tries}: Generating {len(missing)} missing simulations...")
                try:
                    generated = generate_for_params(missing, project_root=project_root, config_path=config_path, n_workers=n_workers)
                    if generated == 0:
                        print(f"Warning: No new simulations generated in attempt {tries + 1}")
                except Exception as e:
                    print(f"Warning: Error during generation attempt {tries + 1}: {e}")
                missing = [p for p in params_list if not _params_to_path(p, timeseries_root).exists()]
                if not missing:
                    break
                tries += 1
            if missing:
                print(f"\nWarning: Could not generate {len(missing)} simulations after {max_tries} attempts")
                print(f"Continuing with {len(params_list) - len(missing)}/{len(params_list)} available simulations")
                params_list = [p for p in params_list if _params_to_path(p, timeseries_root).exists()]
                if len(params_list) == 0:
                    raise RuntimeError("No simulations available. Please check generation parameters and retry.")
                expected_paths = [_params_to_path(p, timeseries_root) for p in params_list]
            pipeline.update(1)

            pipeline.set_postfix_str("torch")
            timeseries_dir = project_root / 'simulation_timeseries'
            torch_dir = project_root / 'torch_data'
            torch_path = build_torch_data(timeseries_dir, torch_dir, max_files=None, sort_by_mtime=False, files=expected_paths)
            pipeline.update(1)

            pipeline.set_postfix_str("processed")
            actual_samples = len([p for p in params_list if _params_to_path(p, timeseries_root).exists()])
            n = build_processed_from_torch_data(output_dir, torch_path, max_samples=actual_samples, config_path=config_path, clear_existing=bool(clear_processed))
            pipeline.update(1)
            return n
        finally:
            pipeline.close()
    else:
        pipeline = tqdm(total=3, desc="Dataset pipeline")
        try:
            pipeline.set_postfix_str("generate")
            generate_simulations(max_new=n_samples, project_root=project_root, config_path=config_path, n_workers=n_workers)
            pipeline.update(1)

            pipeline.set_postfix_str("torch")
            timeseries_dir = project_root / 'simulation_timeseries'
            torch_dir = project_root / 'torch_data'
            torch_path = build_torch_data(timeseries_dir, torch_dir, max_files=n_samples, sort_by_mtime=True)
            pipeline.update(1)

            pipeline.set_postfix_str("processed")
            n = build_processed_from_torch_data(output_dir, torch_path, max_samples=n_samples, config_path=config_path, clear_existing=bool(clear_processed))
            pipeline.update(1)
            return n
        finally:
            pipeline.close()


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 8,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    config_path: Path | None = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    project_root = Path(data_dir).parent.parent
    gen_cfg = load_generation_config(config_path=config_path, project_root=project_root)
    dataset = ProppantDataset(data_dir, gen_cfg=gen_cfg)

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


def build_processed_from_torch_data(
    output_dir: Path,
    torch_data_path: Path,
    max_samples: Optional[int] = None,
    config_path: Path | None = None,
    clear_existing: bool = False
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        old_files = list(output_dir.glob("sample_*.npz"))
        for f in tqdm(old_files, desc="Cleaning processed", leave=False):
            try:
                f.unlink()
            except Exception:
                pass

    torch_data_path = Path(torch_data_path)
    data = torch.load(torch_data_path, map_location='cpu', weights_only=False)

    Q = data['Q']
    times = data['times']
    params_raw = data['params']

    if isinstance(Q, torch.Tensor):
        Q = Q.detach().cpu().numpy()
    if isinstance(times, torch.Tensor):
        times = times.detach().cpu().numpy()
    if isinstance(params_raw, torch.Tensor):
        params_raw = params_raw.detach().cpu().numpy()

    Q = Q.astype(np.float32, copy=False)
    times = times.astype(np.float32, copy=False)
    params_raw = params_raw.astype(np.float32, copy=False)

    n_samples, n_times, ny, nx = Q.shape

    if max_samples is not None:
        n_keep = int(max(0, min(n_samples, max_samples)))
        Q = Q[:n_keep]
        times = times[:n_keep]
        params_raw = params_raw[:n_keep]
        n_samples = n_keep

    if n_samples == 0:
        raise ValueError("No samples selected")

    w0 = params_raw[:, 1].reshape(n_samples, 1, 1, 1)
    w0_safe = np.where(w0 == 0, 1.0, w0)
    concentrations = Q / w0_safe

    project_root = output_dir.parent.parent
    gen_cfg = load_generation_config(config_path=config_path, project_root=project_root)
    cmax = np.float32(gen_cfg.get('physics', {}).get('cmax', 0.635))
    concentrations = np.clip(concentrations, 0.0, float(cmax)).astype(np.float32, copy=False)
    concentrations = (concentrations / cmax).astype(np.float32, copy=False)

    for i in tqdm(range(n_samples), desc="Fixing tails", leave=False):
        t = times[i]
        idx = np.where(t > 0)[0]
        if idx.size == 0:
            last = 0
        else:
            last = int(idx[-1])
        if last < (n_times - 1):
            times[i, last + 1:] = times[i, last]
            concentrations[i, last + 1:] = concentrations[i, last]

    param_names = ['c_in', 'w0', 'mu0', 'Q', 'chi', 'c_in_times', 'dT']
    if params_raw.shape[1] != len(param_names):
        raise ValueError(f"Expected {len(param_names)} params, got {params_raw.shape[1]}")

    p_min = np.zeros((len(param_names),), dtype=np.float32)
    p_max = np.ones((len(param_names),), dtype=np.float32)
    spec = gen_cfg.get('params', {})
    for i, name in enumerate(param_names):
        if name == 'dT':
            vals = spec.get('dT_values', [])
            if isinstance(vals, list) and len(vals) > 0:
                p_min[i] = float(np.min(vals))
                p_max[i] = float(np.max(vals))
            else:
                p_min[i] = float(params_raw[:, i].min())
                p_max[i] = float(params_raw[:, i].max())
            continue
        s = spec.get(name, None)
        if isinstance(s, dict) and 'min' in s and 'max' in s:
            p_min[i] = float(s['min'])
            p_max[i] = float(s['max'])
        else:
            p_min[i] = float(params_raw[:, i].min())
            p_max[i] = float(params_raw[:, i].max())
    denom = (p_max - p_min)
    denom = np.where(denom == 0, 1.0, denom)
    params = (params_raw - p_min) / denom
    params = params.astype(np.float32, copy=False)

    start_idx = 0
    if not clear_existing:
        existing = sorted(output_dir.glob("sample_*.npz"))
        if existing:
            ids = []
            for f in existing:
                stem = f.stem
                if stem.startswith("sample_"):
                    try:
                        ids.append(int(stem.split("_", 1)[1]))
                    except Exception:
                        pass
            if ids:
                start_idx = max(ids) + 1

    meta_path = output_dir / 'metadata.json'
    if meta_path.exists() and not clear_existing:
        try:
            with open(meta_path) as f:
                old_meta = json.load(f)
            if int(old_meta.get('nx', nx)) != int(nx) or int(old_meta.get('ny', ny)) != int(ny) or int(old_meta.get('n_times', n_times)) != int(n_times):
                raise ValueError("Existing processed dataset incompatible. Use --clear-processed.")
        except Exception:
            raise

    for i in tqdm(range(n_samples), desc="Writing samples", leave=False):
        np.savez_compressed(
            output_dir / f"sample_{(start_idx + i):05d}.npz",
            concentrations=concentrations[i],
            times=times[i],
            params=params[i],
            params_raw=params_raw[i],
        )

    total_out = len(list(output_dir.glob("sample_*.npz")))
    metadata = {
        'source': str(torch_data_path),
        'n_samples': int(total_out),
        'nx': int(nx),
        'ny': int(ny),
        'L': float(gen_cfg.get('grid', {}).get('L', 60.0)),
        'H': float(gen_cfg.get('grid', {}).get('H', 60.0)),
        'cmax': float(cmax),
        'n_times': int(n_times),
        'Tmax': float(np.max(times)),
        'dT': float(np.median(np.diff(times[0])) if times.shape[1] > 1 else 0.0),
        'param_names': param_names,
        'param_min': p_min.astype(np.float32).tolist(),
        'param_max': p_max.astype(np.float32).tolist(),
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return n_samples


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
