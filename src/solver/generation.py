import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import hashlib
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

_solver_dir = Path(__file__).parent
if str(_solver_dir) not in sys.path:
    sys.path.append(str(_solver_dir))
from CPU_solver.SystemSolverCPU import SolverCPU

def load_generation_config(config_path: Path | None = None, project_root: Path | None = None) -> dict:
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    if config_path is None:
        config_path = Path(project_root) / 'configs' / 'default.json'
    config_path = Path(config_path)
    cfg = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    gen = cfg.get('solver_generation', {})
    if not gen:
        gen = {
            'seed': 42,
            'grid': {'L': 60.0, 'H': 60.0, 'Nx': 100, 'Ny': 100, 'Tmax': 400.0},
            'params': {
                'c_in': {'min': 0.05, 'max': 0.45, 'step': 0.05},
                'w0': {'min': 0.01, 'max': 0.03, 'step': 0.005},
                'mu0': {'min': 0.001, 'max': 0.1, 'step': 0.005},
                'Q': {'min': -0.5, 'max': -0.01, 'step': 0.05},
                'chi': {'min': 15.0, 'max': 30.0, 'step': 3.0},
                'c_in_times': {'min': 50.0, 'max': 200.0, 'step': 25.0},
                'dT_values': [2.0],
            },
            'numerics': {
                'CFL': 0.8, 'rk_stages': 3, 'lim_type': 'koren', 'kappa': -1,
                'WENO_type': 'Z', 'use_WENO': False, 'prefer_CG': True,
                'prefer_async_CG': True, 'async_check_interval': 1,
                'eps': 1e-10, 'max_iter': 100000
            },
            'physics': {'beta': 2.5, 'mu0': None, 'cmax': 0.635, 'rho1': 1e3, 'rho2': 2.7e3, 'g': 0.0, 'r': 0.0},
            'boundary': {'c_in_times_tail': [150.0], 'c_in_arr_tail': [0.0, 0.1]},
        }
    return gen


def _values_from_spec(spec):
    if spec is None:
        return []
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict) and 'values' in spec:
        return spec['values']
    if isinstance(spec, dict) and all(k in spec for k in ('min', 'max', 'step')):
        mn = float(spec['min'])
        mx = float(spec['max'])
        st = float(spec['step'])
        if st == 0:
            return [mn]
        n = int(np.floor((mx - mn) / st + 1e-9)) + 1
        vals = [mn + i * st for i in range(n)]
        if vals and vals[-1] < mx - 1e-9:
            vals.append(mx)
        return vals
    return []


def build_param_combinations(gen_cfg: dict) -> list[tuple]:
    p = gen_cfg.get('params', {})
    c_in = _values_from_spec(p.get('c_in'))
    w0 = _values_from_spec(p.get('w0'))
    mu0 = _values_from_spec(p.get('mu0'))
    Q = _values_from_spec(p.get('Q'))
    chi = _values_from_spec(p.get('chi'))
    c_in_times = _values_from_spec(p.get('c_in_times'))
    dT_values = p.get('dT_values', [])
    return list(itertools.product(c_in, w0, mu0, Q, chi, c_in_times, dT_values))

def get_param_hash(params):
    param_str = '_'.join(f'{p:.6f}' if isinstance(p, float) else f'{p:.1f}' for p in params)
    return hashlib.md5(param_str.encode()).hexdigest()

def get_gen_hash(gen_cfg: dict) -> str:
    grid = gen_cfg.get('grid', {})
    numerics = gen_cfg.get('numerics', {})
    physics = gen_cfg.get('physics', {})
    boundary = gen_cfg.get('boundary', {})
    payload = {
        'grid': {
            'L': float(grid.get('L', 0.0)),
            'H': float(grid.get('H', 0.0)),
            'Nx': int(grid.get('Nx', 0)),
            'Ny': int(grid.get('Ny', 0)),
            'Tmax': float(grid.get('Tmax', 0.0)),
        },
        'numerics': {
            'CFL': float(numerics.get('CFL', 0.0)),
            'rk_stages': int(numerics.get('rk_stages', 0)),
            'lim_type': str(numerics.get('lim_type', '')),
            'kappa': float(numerics.get('kappa', 0.0)),
            'WENO_type': str(numerics.get('WENO_type', '')),
            'use_WENO': bool(numerics.get('use_WENO', False)),
            'prefer_CG': bool(numerics.get('prefer_CG', False)),
            'prefer_async_CG': bool(numerics.get('prefer_async_CG', False)),
            'async_check_interval': int(numerics.get('async_check_interval', 0)),
            'eps': float(numerics.get('eps', 0.0)),
            'max_iter': int(numerics.get('max_iter', 0)),
        },
        'physics': {
            'beta': float(physics.get('beta', 0.0)),
            'cmax': float(physics.get('cmax', 0.0)),
            'rho1': float(physics.get('rho1', 0.0)),
            'rho2': float(physics.get('rho2', 0.0)),
            'g': float(physics.get('g', 0.0)),
            'r': float(physics.get('r', 0.0)),
        },
        'boundary': {
            'c_in_times_tail': [float(v) for v in boundary.get('c_in_times_tail', [])],
            'c_in_arr_tail': [float(v) for v in boundary.get('c_in_arr_tail', [])],
        },
    }
    s = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(s.encode()).hexdigest()[:10]

def run_simulation(params, gen_cfg: dict):
    c_in_val, w0_val, mu0_val, Q_val, chi_val, c_in_times_val, dT_val = params

    grid = gen_cfg.get('grid', {})
    L = float(grid.get('L', 60.0))
    H = float(grid.get('H', 60.0))
    Nx = int(grid.get('Nx', 100))
    Ny = int(grid.get('Ny', 100))
    Tmax = float(grid.get('Tmax', 400.0))
    x = np.linspace(0, L, Nx, endpoint=False) + L/Nx/2
    y = np.linspace(0, H, Ny, endpoint=False) + H/Ny/2

    q_in = -Q_val/chi_val * np.where(np.abs(y-H/2) < chi_val/2, 1, 0)
    q_in = (q_in + q_in[::-1]) / 2
    q_out = -np.mean(q_in)

    numerics = gen_cfg.get('numerics', {})
    physics = gen_cfg.get('physics', {})
    boundary = gen_cfg.get('boundary', {})
    c_in_times_tail = boundary.get('c_in_times_tail', [])
    c_in_arr_tail = boundary.get('c_in_arr_tail', [])
    c_in_times = np.array([float(c_in_times_val)] + [float(v) for v in c_in_times_tail], dtype=float)
    c_in_arr = np.array([float(c_in_val)] + [float(v) for v in c_in_arr_tail], dtype=float)
    if c_in_arr.shape[0] != c_in_times.shape[0] + 1:
        raise ValueError(f"Invalid inlet schedule: len(c_in_arr)={c_in_arr.shape[0]} must equal len(c_in_times)+1={c_in_times.shape[0] + 1}")

    config = {
        'grid': {'Tmax': Tmax, 'Ny': Ny, 'Nx': Nx, 'L': L, 'H': H, 'dT': dT_val},
        'physics': {
            'c0': np.zeros((Ny, Nx)),
            'w': w0_val * np.ones((Ny, Nx)),
            'beta': float(physics.get('beta', 2.5)),
            'mu0': mu0_val,
            'cmax': float(physics.get('cmax', 0.635)),
            'rho1': float(physics.get('rho1', 1e3)),
            'rho2': float(physics.get('rho2', 2.7e3)),
            'g': float(physics.get('g', 0.0)),
            'r': float(physics.get('r', 0.0))
        },
        'numerics': {
            'CFL': float(numerics.get('CFL', 0.8)),
            'rk_stages': int(numerics.get('rk_stages', 3)),
            'lim_type': str(numerics.get('lim_type', 'koren')),
            'kappa': float(numerics.get('kappa', -1)),
            'WENO_type': str(numerics.get('WENO_type', 'Z')),
            'use_WENO': bool(numerics.get('use_WENO', False)),
            'prefer_CG': bool(numerics.get('prefer_CG', True)),
            'prefer_async_CG': bool(numerics.get('prefer_async_CG', True)),
            'async_check_interval': int(numerics.get('async_check_interval', 1)),
            'eps': float(numerics.get('eps', 1e-10)),
            'max_iter': int(numerics.get('max_iter', 100000)),
        },
        'boundary_conditions': {
            'q_in': q_in, 'q_out': q_out,
            'c_in_times': c_in_times,
            'c_in_arr': c_in_arr
        }
    }

    s = SolverCPU(config)
    s.solve(s.Tmax)

    time_series_data = []
    time_stamps = []

    if s.step > 0:
        total_steps = s.step + 1
        frame_interval = 1

        for i in range(0, total_steps, frame_interval):
            time_series_data.append(s.Q[i].copy())
            time_stamps.append(s.times[i])

        if (total_steps - 1) % frame_interval != 0:
            last_idx = total_steps - 1
            time_series_data.append(s.Q[last_idx].copy())
            time_stamps.append(s.times[last_idx])

    if len(time_series_data) == 0:
        time_series_data.append(np.zeros((Ny, Nx)))
        time_stamps.append(0.0)

    return s, time_series_data, time_stamps

def _simulate_and_persist(params, gen_cfg: dict, project_root: str, gen_hash: str):
    cwd = os.getcwd()
    os.chdir(str(project_root))
    try:
        param_hash = get_param_hash(params)
        s, time_series_data, time_stamps = run_simulation(params, gen_cfg)
        param_str = f"c{params[0]:.3f}_w{params[1]:.3f}_mu{params[2]:.3f}_Q{params[3]:.3f}_chi{params[4]:.1f}_t{params[5]:.0f}_dT{params[6]:.1f}"
        ts_path = save_time_series(time_series_data, time_stamps, param_str)

        time_metrics = calculate_metrics(time_series_data, s.w)
        time_stats = compute_time_averages(time_metrics) if time_metrics else {}

        final_frame = time_series_data[-1]
        if np.max(s.w) > 0:
            final_c = final_frame / s.w
        else:
            final_c = final_frame
        final_metrics = {
            'final_mean': np.mean(final_c),
            'final_std': np.std(final_c),
            'final_max': np.max(final_c),
            'final_min': np.min(final_c),
            'final_area_above_0.1': np.sum(final_c > 0.1) / final_c.size * 100
        }

        matrix_path = f"simulation_data/{param_str}_final.npy"
        np.save(matrix_path, final_frame)

        result = {
            'c_in': params[0], 'w0': params[1], 'mu0': params[2],
            'Q': params[3], 'chi': params[4], 'c_in_times': params[5],
            'dT': params[6], 'param_hash': param_hash,
            'gen_hash': gen_hash,
            'matrix_path': matrix_path, 'timeseries_path': ts_path,
            'total_steps': len(time_series_data),
            'frames_count': len(time_series_data),
            'max_time': time_stamps[-1] if time_stamps else 0.0
        }
        result.update(time_stats)
        result.update(final_metrics)
        return result
    finally:
        os.chdir(cwd)

def save_time_series(time_series_data, time_stamps, param_str):
    ts_path = f'simulation_timeseries/{param_str}_series.npz'
    Q_array = np.array(time_series_data)
    time_array = np.array(time_stamps)
    np.savez_compressed(ts_path, Q=Q_array, times=time_array)
    return ts_path

def calculate_metrics(time_series_data, w):
    if len(time_series_data) == 0:
        return []

    time_metrics = []
    for Q_frame in time_series_data:
        if np.max(w) > 0:
            c_frame = Q_frame / w
        else:
            c_frame = Q_frame
        metrics = {
            'mean': np.mean(c_frame),
            'std': np.std(c_frame),
            'max': np.max(c_frame),
            'min': np.min(c_frame),
            'area_above_0.1': np.sum(c_frame > 0.1) / c_frame.size * 100
        }
        time_metrics.append(metrics)
    return time_metrics

def compute_time_averages(time_metrics):
    if len(time_metrics) == 0:
        return {}

    avg_metrics = {}
    for key in time_metrics[0].keys():
        values = [m[key] for m in time_metrics]
        avg_metrics[f'time_avg_{key}'] = np.mean(values)
        avg_metrics[f'time_std_{key}'] = np.std(values)
    return avg_metrics

def generate_simulations(max_new: int | None = None, project_root: Path | None = None, config_path: Path | None = None, n_workers: int = 1) -> int:
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    project_root = Path(project_root)
    gen_cfg = load_generation_config(config_path=config_path, project_root=project_root)
    param_combinations = build_param_combinations(gen_cfg)

    sim_data_dir = project_root / 'simulation_data'
    sim_ts_dir = project_root / 'simulation_timeseries'
    sim_data_dir.mkdir(parents=True, exist_ok=True)
    sim_ts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = project_root / 'simulation_results.csv'

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing.dropna(subset=['param_hash'])
        existing_hashes = set(df_existing['param_hash'].tolist())
    else:
        df_existing = pd.DataFrame()
        existing_hashes = set()

    new_simulations = []
    tmax_target = float(gen_cfg.get('grid', {}).get('Tmax', 0.0))
    gen_hash = get_gen_hash(gen_cfg)
    if 'max_time' in df_existing.columns and tmax_target > 0:
        max_time_by_hash = dict(zip(df_existing['param_hash'], df_existing['max_time']))
    else:
        max_time_by_hash = {}
    if 'gen_hash' in df_existing.columns:
        gen_hash_by_hash = dict(zip(df_existing['param_hash'], df_existing['gen_hash']))
    else:
        gen_hash_by_hash = {}
    for params in param_combinations:
        param_hash = get_param_hash(params)
        mt = max_time_by_hash.get(param_hash, None)
        gh = gen_hash_by_hash.get(param_hash, None)
        stale = ((mt is not None) and (float(mt) < tmax_target - 1e-6)) or ((gh is not None) and (str(gh) != gen_hash))
        if (param_hash not in existing_hashes) or stale:
            new_simulations.append(params)

    if max_new is not None:
        seed = int(gen_cfg.get('seed', 42))
        rng = np.random.default_rng(seed)
        if len(new_simulations) > max_new:
            idx = rng.choice(len(new_simulations), size=int(max_new), replace=False)
            new_simulations = [new_simulations[i] for i in idx]

    total_new = len(new_simulations)
    if total_new == 0:
        return 0

    new_results = []
    errors = []
    if n_workers is None or int(n_workers) <= 1:
        for params in tqdm(new_simulations, desc="Generating simulations", total=total_new):
            try:
                result = _simulate_and_persist(params, gen_cfg, str(project_root), gen_hash)
                if result is None:
                    continue
                new_results.append(result)
                df_new = pd.DataFrame(new_results)
                if not df_existing.empty and 'param_hash' in df_existing.columns:
                    df_existing = df_existing[df_existing['param_hash'] != result['param_hash']]
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(csv_path, index=False)
            except Exception as e:
                errors.append(str(e))
                if len(errors) <= 3:
                    print(f"\n[ERROR] Simulation failed: {e}")
                continue
    else:
        max_workers = int(n_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_simulate_and_persist, params, gen_cfg, str(project_root), gen_hash) for params in new_simulations]
            for fut in tqdm(as_completed(futures), total=total_new, desc="Generating simulations"):
                try:
                    result = fut.result()
                    if result is None:
                        continue
                    new_results.append(result)
                    df_new = pd.DataFrame(new_results)
                    if not df_existing.empty and 'param_hash' in df_existing.columns:
                        df_existing = df_existing[df_existing['param_hash'] != result['param_hash']]
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(csv_path, index=False)
                except Exception as e:
                    errors.append(str(e))
                    if len(errors) <= 3:
                        print(f"\n[ERROR] Simulation failed: {e}")
                    continue

    if errors:
        print(f"\n[WARNING] {len(errors)}/{total_new} simulations failed")

    return len(new_results)


def generate_for_params(params_list: list[tuple], project_root: Path | None = None, config_path: Path | None = None, n_workers: int = 1) -> int:
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    project_root = Path(project_root)
    gen_cfg = load_generation_config(config_path=config_path, project_root=project_root)

    sim_data_dir = project_root / 'simulation_data'
    sim_ts_dir = project_root / 'simulation_timeseries'
    sim_data_dir.mkdir(parents=True, exist_ok=True)
    sim_ts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = project_root / 'simulation_results.csv'

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing.dropna(subset=['param_hash'])
        existing_hashes = set(df_existing['param_hash'].tolist())
    else:
        df_existing = pd.DataFrame()
        existing_hashes = set()

    params_list = list(params_list)
    tmax_target = float(gen_cfg.get('grid', {}).get('Tmax', 0.0))
    gen_hash = get_gen_hash(gen_cfg)
    if 'max_time' in df_existing.columns and tmax_target > 0:
        max_time_by_hash = dict(zip(df_existing['param_hash'], df_existing['max_time']))
    else:
        max_time_by_hash = {}
    if 'gen_hash' in df_existing.columns:
        gen_hash_by_hash = dict(zip(df_existing['param_hash'], df_existing['gen_hash']))
    else:
        gen_hash_by_hash = {}
    filtered = []
    for p in params_list:
        h = get_param_hash(p)
        mt = max_time_by_hash.get(h, None)
        gh = gen_hash_by_hash.get(h, None)
        stale = ((mt is not None) and (float(mt) < tmax_target - 1e-6)) or ((gh is not None) and (str(gh) != gen_hash))
        if (h not in existing_hashes) or stale:
            filtered.append(p)
    params_list = filtered
    if len(params_list) == 0:
        return 0
    new_results = []
    errors = []
    if n_workers is None or int(n_workers) <= 1:
        for params in tqdm(params_list, desc="Generating simulations", total=len(params_list)):
            try:
                result = _simulate_and_persist(params, gen_cfg, str(project_root), gen_hash)
                if result is None:
                    continue
                new_results.append(result)
                df_new = pd.DataFrame(new_results)
                if not df_existing.empty and 'param_hash' in df_existing.columns:
                    df_existing = df_existing[df_existing['param_hash'] != result['param_hash']]
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(csv_path, index=False)
            except Exception as e:
                errors.append(f"params={params[:3]}...: {e}")
                if len(errors) <= 3:
                    print(f"\n[ERROR] Simulation failed: {e}")
                continue
    else:
        max_workers = int(n_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_simulate_and_persist, params, gen_cfg, str(project_root), gen_hash) for params in params_list]
            for fut in tqdm(as_completed(futures), total=len(params_list), desc="Generating simulations"):
                try:
                    result = fut.result()
                    if result is None:
                        continue
                    new_results.append(result)
                    df_new = pd.DataFrame(new_results)
                    if not df_existing.empty and 'param_hash' in df_existing.columns:
                        df_existing = df_existing[df_existing['param_hash'] != result['param_hash']]
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(csv_path, index=False)
                except Exception as e:
                    errors.append(f"{e}")
                    if len(errors) <= 3:
                        print(f"\n[ERROR] Simulation failed: {e}")
                    continue

    if errors:
        print(f"\n[WARNING] {len(errors)} simulations failed. First error: {errors[0]}")

    return len(new_results)


if __name__ == '__main__':
    n = generate_simulations()
    print(f"Generated {n} simulations")