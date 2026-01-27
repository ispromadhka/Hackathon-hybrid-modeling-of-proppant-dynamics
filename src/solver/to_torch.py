import numpy as np
import torch
from pathlib import Path
import re
from tqdm import tqdm

def parse_filename(filename):
    pattern = r'c([\d.]+)_w([\d.]+)_mu([\d.]+)_Q([-\d.]+)_chi([\d.]+)_t(\d+)_dT([\d.]+)_series\.npz'
    match = re.match(pattern, filename)
    if match:
        return {
            'c_in': float(match.group(1)),
            'w0': float(match.group(2)),
            'mu0': float(match.group(3)),
            'Q': float(match.group(4)),
            'chi': float(match.group(5)),
            'c_in_times': int(match.group(6)),
            'dT': float(match.group(7))
        }
    return None

def load_timeseries(filepath):
    data = np.load(filepath, allow_pickle=True)
    Q = data['Q']
    times = data['times']
    return Q, times

def build_torch_data(timeseries_dir: Path, output_dir: Path, max_files: int | None = None, sort_by_mtime: bool = False, files: list[Path] | None = None) -> Path:
    timeseries_dir = Path(timeseries_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if files is None:
        files = list(timeseries_dir.glob('*_series.npz'))
        if sort_by_mtime:
            files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        else:
            files = sorted(files)
    else:
        files = [Path(p) for p in files]
    valid_files = []
    valid_params = []

    for file in tqdm(files, desc="Parsing files"):
        params = parse_filename(file.name)
        if params:
            valid_files.append(file)
            valid_params.append(params)

    if max_files is not None:
        valid_files = valid_files[:max_files]
        valid_params = valid_params[:max_files]

    if len(valid_files) == 0:
        raise ValueError("No simulation files found")

    Q_list = []
    times_list = []
    params_list = []
    ny = None
    nx = None

    for file, params in tqdm(zip(valid_files, valid_params), total=len(valid_files), desc="Loading data"):
        Q, times = load_timeseries(file)
        if Q.ndim != 3:
            continue
        if ny is None or nx is None:
            ny, nx = int(Q.shape[1]), int(Q.shape[2])
        if int(Q.shape[1]) != int(ny) or int(Q.shape[2]) != int(nx):
            continue
        Q_list.append(Q)
        times_list.append(times)
        params_array = np.array([
            params['c_in'],
            params['w0'],
            params['mu0'],
            params['Q'],
            params['chi'],
            params['c_in_times'],
            params['dT']
        ], dtype=np.float32)
        params_list.append(params_array)

    if len(Q_list) == 0:
        raise ValueError("No compatible simulation files found")

    max_frames = max(Q.shape[0] for Q in Q_list)
    n_samples = len(Q_list)
    n_params = 7

    Q_padded = np.zeros((n_samples, max_frames, ny, nx), dtype=np.float32)
    times_padded = np.zeros((n_samples, max_frames), dtype=np.float32)
    params_array = np.array(params_list, dtype=np.float32)

    for i, (Q, times) in enumerate(zip(Q_list, times_list)):
        n_frames = Q.shape[0]
        Q_padded[i, :n_frames] = Q
        times_padded[i, :n_frames] = times

    data = {
        'Q': torch.from_numpy(Q_padded),
        'times': torch.from_numpy(times_padded),
        'params': torch.from_numpy(params_array),
        'n_samples': n_samples,
        'n_frames': max_frames,
        'ny': ny,
        'nx': nx,
        'n_params': n_params
    }

    output_path = output_dir / 'data.pt'
    torch.save(data, output_path)

    npz_path = output_dir / 'data.npz'
    np.savez_compressed(
        npz_path,
        Q=Q_padded,
        times=times_padded,
        params=params_array,
        n_samples=n_samples,
        n_frames=max_frames,
        ny=ny,
        nx=nx,
        n_params=n_params
    )

    return output_path


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent
    timeseries_dir = project_root / 'simulation_timeseries'
    output_dir = project_root / 'torch_data'
    out = build_torch_data(timeseries_dir, output_dir)
    y = torch.load(out, map_location='cpu', weights_only=False)
    print(f"Shape of Q: {tuple(y['Q'].shape)}")
    print(f"Shape of times: {tuple(y['times'].shape)}")
    print(f"Shape of params: {tuple(y['params'].shape)}")

