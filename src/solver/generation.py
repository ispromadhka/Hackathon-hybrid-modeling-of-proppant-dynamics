import os
import sys
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import itertools
import hashlib
import warnings
warnings.filterwarnings('ignore')

module_path = os.path.abspath('../')
if module_path not in sys.path:
    sys.path.append(module_path)
from CPU_solver.SystemSolverCPU import SolverCPU

os.makedirs('simulation_data', exist_ok=True)
os.makedirs('simulation_timeseries', exist_ok=True)
csv_path = 'simulation_results.csv'

if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path)
    existing_hashes = set(df_existing['param_hash'].tolist())
else:
    df_existing = pd.DataFrame()
    existing_hashes = set()

L, H = 60, 60
Nx = Ny = 100
Tmax = 400

c_in = [0.05]
w0 = [0.010]
mu0 = [0.001]
Q = [-0.05]
chi = [H/6]
c_in_times_list = [50]
dT_values = [2]

param_combinations = list(itertools.product(c_in, w0, mu0, Q, chi, c_in_times_list, dT_values))

def get_param_hash(params):
    param_str = '_'.join(f'{p:.6f}' if isinstance(p, float) else f'{p:.1f}' for p in params)
    return hashlib.md5(param_str.encode()).hexdigest()

def run_simulation(params):
    c_in_val, w0_val, mu0_val, Q_val, chi_val, c_in_times_val, dT_val = params
    
    x = np.linspace(0, L, Nx, endpoint=False) + L/Nx/2
    y = np.linspace(0, H, Ny, endpoint=False) + H/Ny/2
    
    q_in = -Q_val/chi_val * np.where(np.abs(y-H/2) < chi_val/2, 1, 0)
    q_in = (q_in + q_in[::-1]) / 2
    q_out = -np.mean(q_in)
    
    config = {
        'grid': {'Tmax': Tmax, 'Ny': Ny, 'Nx': Nx, 'L': L, 'H': H, 'dT': dT_val},
        'physics': {
            'c0': np.zeros((Nx, Ny)),
            'w': w0_val * np.ones((Nx, Ny)),
            'beta': 2.5,
            'mu0': mu0_val,
            'cmax': 0.635,
            'rho1': 1e3,
            'rho2': 2.7e3,
            'g': 0,
            'r': 0
        },
        'numerics': {
            'CFL': 0.8, 'rk_stages': 1, 'lim_type': 'koren', 'kappa': -1,
            'WENO_type': 'Z', 'use_WENO': False, 'prefer_CG': True,
            'prefer_async_CG': True, 'async_check_interval': 1,
            'eps': 1e-10, 'max_iter': 100000
        },
        'boundary_conditions': {
            'q_in': q_in, 'q_out': q_out,
            'c_in_times': np.array([c_in_times_val, 150]),
            'c_in_arr': np.array([c_in_val, 0.0, 0.1])
        }
    }
    
    s = SolverCPU(config)
    s.solve(s.Tmax)
    
    time_series_data = []
    time_stamps = []
    
    if s.step > 0:
        total_steps = s.step + 1
        frame_interval = max(1, total_steps // 100)
        
        for i in range(0, total_steps, frame_interval):
            if i < len(s.Q):
                time_series_data.append(s.Q[i].copy())
                time_stamps.append(s.times[i])
        
        if total_steps > 0 and (total_steps - 1) % frame_interval != 0:
            last_idx = min(total_steps - 1, len(s.Q) - 1)
            time_series_data.append(s.Q[last_idx].copy())
            time_stamps.append(s.times[last_idx])
    
    if len(time_series_data) == 0:
        time_series_data.append(np.zeros((Nx, Ny)))
        time_stamps.append(0.0)
    
    return s, time_series_data, time_stamps

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

new_results = []

for params in param_combinations:
    param_hash = get_param_hash(params)
    
    if param_hash in existing_hashes:
        continue
    
    try:
        s, time_series_data, time_stamps = run_simulation(params)
        
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
        
        matrix_path = f'simulation_data/{param_str}_final.npy'
        np.save(matrix_path, final_frame)
        
        result = {
            'c_in': params[0], 'w0': params[1], 'mu0': params[2],
            'Q': params[3], 'chi': params[4], 'c_in_times': params[5],
            'dT': params[6], 'param_hash': param_hash,
            'matrix_path': matrix_path, 'timeseries_path': ts_path,
            'total_steps': len(time_series_data),
            'frames_count': len(time_series_data),
            'max_time': time_stamps[-1] if time_stamps else 0.0
        }
        
        result.update(time_stats)
        result.update(final_metrics)
        new_results.append(result)
        
        df_new = pd.DataFrame(new_results)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
            
    except Exception as e:
        print(f"Failed for params {params}: {str(e)}")
        continue

if new_results:
    df_final = pd.concat([df_existing, pd.DataFrame(new_results)], ignore_index=True)
    df_final.to_csv(csv_path, index=False)
    print(f"Added {len(new_results)} new simulations. Total: {len(df_final)}")
else:
    print("No new simulations to run.")
    df_final = df_existing

print(f"Dataset contains {len(df_final)} simulations")
print(df_final[['c_in', 'w0', 'mu0', 'Q', 'chi', 'c_in_times', 'dT', 'frames_count', 'max_time']].head())
print("Process completed successfully")