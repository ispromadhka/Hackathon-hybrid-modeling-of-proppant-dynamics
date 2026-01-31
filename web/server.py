"""
FastAPI backend for Proppant Transport Simulator.
Supports both old (complex weights) and new model formats.
"""

import json
import time
from pathlib import Path
from typing import Optional
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.fno import create_model
from src.solver.solver_wrapper import ProppantSolver

# Paths
ROOT = Path(__file__).parent.parent
WEB_DIR = Path(__file__).parent
CHECKPOINT_DIR = ROOT / 'checkpoints'
# Try adaptive first, then classic
CHECKPOINT_PATH_ADAPTIVE = CHECKPOINT_DIR / 'adaptive' / 'best.pt'
CHECKPOINT_PATH_CLASSIC = CHECKPOINT_DIR / 'classic' / 'best.pt'
CHECKPOINT_PATH = CHECKPOINT_PATH_ADAPTIVE if CHECKPOINT_PATH_ADAPTIVE.exists() else CHECKPOINT_PATH_CLASSIC
DATA_META_PATH = ROOT / 'data' / 'processed' / 'metadata.json'

# Global model state
MODEL = None
MODEL_META = None
MODEL_VALID = False
MODEL_WARNING = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app = FastAPI(title="Proppant Transport Simulator")

# Mount static files
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


class SimulationParams(BaseModel):
    c_in: float = 0.3        # Training range: 0.05 - 0.45
    w0: float = 0.015        # Training range: 0.01 - 0.03
    mu0: float = 0.001       # Training range: 0.001 - 0.02
    Q: float = 0.05          # Training range: 0.01 - 0.2 (becomes negative internally)
    chi: float = 20.0        # Training range: 15 - 30 (inlet width in meters)
    c_in_times: float = 100.0  # Training range: 50 - 200


def load_model():
    """Load the trained model with compatibility for old complex weights."""
    global MODEL, MODEL_META, MODEL_VALID, MODEL_WARNING

    MODEL_VALID = False
    MODEL_WARNING = None

    # Try to find checkpoint in adaptive or classic subdirectories
    checkpoint_path = None
    smoothing_type = None

    # Try adaptive first
    if CHECKPOINT_PATH_ADAPTIVE.exists():
        checkpoint_path = CHECKPOINT_PATH_ADAPTIVE
        smoothing_type = 'adaptive'
    elif (CHECKPOINT_DIR / 'adaptive' / 'last.pt').exists():
        checkpoint_path = CHECKPOINT_DIR / 'adaptive' / 'last.pt'
        smoothing_type = 'adaptive'
        print(f"best.pt не найден в adaptive/, используем last.pt")
    # Try classic
    elif CHECKPOINT_PATH_CLASSIC.exists():
        checkpoint_path = CHECKPOINT_PATH_CLASSIC
        smoothing_type = 'classic'
    elif (CHECKPOINT_DIR / 'classic' / 'last.pt').exists():
        checkpoint_path = CHECKPOINT_DIR / 'classic' / 'last.pt'
        smoothing_type = 'classic'
        print(f"best.pt не найден в classic/, используем last.pt")
    # Fallback to old location (root checkpoints/)
    elif (CHECKPOINT_DIR / 'best.pt').exists():
        checkpoint_path = CHECKPOINT_DIR / 'best.pt'
        smoothing_type = 'classic'  # Assume classic for old checkpoints
        print(f"Используется старый формат чекпоинта из корня checkpoints/")
    elif (CHECKPOINT_DIR / 'last.pt').exists():
        checkpoint_path = CHECKPOINT_DIR / 'last.pt'
        smoothing_type = 'classic'
        print(f"Используется старый формат чекпоинта из корня checkpoints/")
    else:
        MODEL_WARNING = f"Чекпоинт не найден в {CHECKPOINT_DIR}/adaptive/ или {CHECKPOINT_DIR}/classic/"
        print(f"Предупреждение: {MODEL_WARNING}")
        return False

    # Load metadata
    if DATA_META_PATH.exists():
        with open(DATA_META_PATH) as f:
            MODEL_META = json.load(f)
    else:
        # Defaults must match configs/default.json solver_generation.grid
        MODEL_META = {
            'nx': 150,
            'ny': 100,
            'n_times': 201,
            'param_names': ['c_in', 'w0', 'mu0', 'Q', 'chi', 'c_in_times', 'dT'],
            'param_min': [0.05, 0.01, 0.001, -0.2, 15.0, 50.0, 2.0],
            'param_max': [0.45, 0.03, 0.02, -0.01, 30.0, 200.0, 2.0],
            'cmax': 0.635,
            'L': 100.0,
            'H': 60.0,
            'Tmax': 400.0,
            'dT': 2.0
        }

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if 'base_model_state_dict' in checkpoint:
            state_dict = checkpoint['base_model_state_dict']
        else:
            state_dict = checkpoint.get('model_state_dict', checkpoint)
    except Exception as e:
        MODEL_WARNING = f"Ошибка загрузки checkpoint: {e}"
        print(f"Предупреждение: {MODEL_WARNING}")
        return False

    # Detect model dimensions from checkpoint
    if 'grid_x' in state_dict:
        nx = state_dict['grid_x'].shape[2]
        ny = state_dict['grid_x'].shape[3]
    else:
        nx = MODEL_META.get('nx', 100)
        ny = MODEL_META.get('ny', 100)

    if 'lift.0.weight' in state_dict:
        n_params = state_dict['lift.0.weight'].shape[1] - 2
    elif 'lift.weight' in state_dict:
        n_params = state_dict['lift.weight'].shape[1] - 2
    else:
        n_params = len(MODEL_META.get('param_names', []))

    n_times = None
    if 'project.6.weight' in state_dict:
        n_times = state_dict['project.6.weight'].shape[0]
    elif 'project.5.weight' in state_dict:
        n_times = state_dict['project.5.weight'].shape[0]
    elif 'project.2.weight' in state_dict:
        n_times = state_dict['project.2.weight'].shape[0]
    elif 'spatial_smooth.kernel' in state_dict:
        n_times = state_dict['spatial_smooth.kernel'].shape[0]
    else:
        n_times = MODEL_META.get('n_times', 201)

    print(f"Создание модели НС: nx={nx}, ny={ny}, n_times={n_times}, n_params={n_params}")

    # Detect if checkpoint uses adaptive smoothing
    # First check by directory, then by state_dict keys
    if smoothing_type:
        use_adaptive = (smoothing_type == 'adaptive')
    else:
        has_adaptive = 'adaptive_temporal_smooth.kernel' in state_dict
        has_old_smoothing = 'spatial_smooth.kernel' in state_dict or any('lowpass' in k for k in state_dict.keys())
        use_adaptive = has_adaptive and not has_old_smoothing

    if use_adaptive:
        print("Используется адаптивное временное сглаживание")
    else:
        print("Используется классическое сглаживание (lowpass + spatial + temporal)")

    MODEL = create_model(
        nx=nx, ny=ny, n_times=n_times, n_params=n_params, device=DEVICE,
        model_cfg={'use_error_corrector': False, 'use_adaptive_smoothing': use_adaptive}
    )

    # Filter incompatible weights
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if 'error_corrector' in k:
            continue
        if use_adaptive and ('spatial_smooth' in k or 'temporal_smooth' in k or 'lowpass' in k):
            continue
        if not use_adaptive and 'adaptive_temporal_smooth' in k:
            continue
        filtered_state_dict[k] = v

    try:
        missing, unexpected = MODEL.load_state_dict(filtered_state_dict, strict=False)

        total_params = len(list(MODEL.state_dict().keys()))
        missing_ratio = len(missing) / total_params if total_params > 0 else 1.0

        if missing_ratio > 0.3:
            MODEL_WARNING = f"Несовместимость архитектуры НС: {len(missing)}/{total_params} весов не загружено. Требуется переобучение модели."
            print(f"ВНИМАНИЕ: {MODEL_WARNING}")
            MODEL_VALID = False
        elif missing:
            MODEL_WARNING = f"Частичная загрузка НС: {len(missing)} весов отсутствует"
            print(f"Предупреждение: {MODEL_WARNING}")
            MODEL_VALID = True
        else:
            MODEL_VALID = True
            print(f"Модель НС загружена успешно — все веса совпали")

        if unexpected:
            print(f"Примечание: {len(unexpected)} лишних ключей проигнорировано")

    except Exception as e:
        MODEL_WARNING = f"Ошибка загрузки весов НС: {e}"
        print(f"Предупреждение: {MODEL_WARNING}")
        MODEL_VALID = False

    MODEL.eval()
    return MODEL_VALID


def normalize_params(raw: np.ndarray) -> np.ndarray:
    """Normalize parameters to [0, 1] range."""
    if MODEL_META is None:
        return raw.astype(np.float32)

    pmin = np.array(MODEL_META.get('param_min', [0] * len(raw)), dtype=np.float32)
    pmax = np.array(MODEL_META.get('param_max', [1] * len(raw)), dtype=np.float32)
    denom = pmax - pmin
    denom = np.where(denom == 0, 1.0, denom)
    x = (raw.astype(np.float32) - pmin) / denom
    return np.clip(x, 0.0, 1.0).astype(np.float32)


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    load_model()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    html_path = WEB_DIR / "templates" / "index.html"
    return FileResponse(html_path)


@app.post("/api/simulate")
async def simulate(params: SimulationParams):
    """Run simulation with both NN and NS solver."""

    # Get config values
    dT = MODEL_META.get('dT', 2.0) if MODEL_META else 2.0
    Tmax = MODEL_META.get('Tmax', 400.0) if MODEL_META else 400.0
    cmax = MODEL_META.get('cmax', 0.635) if MODEL_META else 0.635
    Lx = MODEL_META.get('L', 60.0) if MODEL_META else 60.0
    Ly = MODEL_META.get('H', 60.0) if MODEL_META else 60.0
    nx = MODEL_META.get('nx', 100) if MODEL_META else 100
    ny = MODEL_META.get('ny', 100) if MODEL_META else 100

    # Note: Negative Q for solver convention (inflow)
    Q_internal = -abs(params.Q)
    print(f"[DEBUG] Input params: c_in={params.c_in}, w0={params.w0}, mu0={params.mu0}, Q={params.Q}, chi={params.chi}")

    raw_params = np.array([
        params.c_in,
        params.w0,
        params.mu0,
        Q_internal,
        params.chi,
        params.c_in_times,
        dT
    ], dtype=np.float32)

    # ===== Neural Network Prediction =====
    nn_result = None
    nn_time = 0.0
    nn_available = False

    if MODEL is not None and MODEL_VALID:
        try:
            norm_params = normalize_params(raw_params)
            inp = torch.from_numpy(norm_params.reshape(1, -1)).to(DEVICE).float()

            t0 = time.perf_counter()
            with torch.no_grad():
                pred = MODEL(inp)[0].cpu().numpy()
            nn_time = time.perf_counter() - t0

            pred = pred.transpose(0, 2, 1)
            nn_result = pred * cmax
            nn_available = True

        except Exception as e:
            print(f"NN prediction error: {e}")
            nn_result = None
            nn_available = False

    # ===== Numerical Solver =====
    ns_result = None
    ns_time = 0.0
    times = None

    try:
        c_in_times_arr = np.array([params.c_in_times])
        c_in_arr = np.array([params.c_in, 0.0])

        # Convert chi (inlet width in meters) to inlet_fraction
        inlet_fraction = min(params.chi / Ly, 1.0)
        print(f"[DEBUG] chi={params.chi}, Ly={Ly}, inlet_fraction={inlet_fraction:.4f}")

        solver = ProppantSolver(
            nx=nx, ny=ny,
            Lx=Lx, Ly=Ly,
            T=Tmax, dT=dT,
            c_inlet=params.c_in,
            Q_inlet=Q_internal,
            g=0.0,  # Гравитация = 0 (как в обучающих данных)
            mu0=params.mu0,
            r_particle=0.0,  # r=0 (как в обучающих данных)
            w0=params.w0,
            inlet_fraction=inlet_fraction,
            c_in_times=c_in_times_arr,
            c_in_arr=c_in_arr,
            rk_stages=3,
            lim_type='koren',
        )

        t0 = time.perf_counter()
        times, concentrations = solver.solve()
        ns_time = time.perf_counter() - t0

        # Debug: log concentration statistics
        print(f"[DEBUG] Solver output shape: {concentrations.shape}")
        print(f"[DEBUG] Concentration range: [{concentrations.min():.6f}, {concentrations.max():.6f}]")
        print(f"[DEBUG] Non-zero elements: {np.count_nonzero(concentrations)} / {concentrations.size}")
        print(f"[DEBUG] Times: {len(times)} steps, range [{times[0]:.1f}, {times[-1]:.1f}]")

        ns_result = np.clip(concentrations, 0, cmax)

    except Exception as e:
        print(f"NS solver error: {e}")
        raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")

    if ns_result is None:
        raise HTTPException(status_code=500, detail="Solver failed")

    # Match frame counts and spatial dimensions
    if nn_result is not None and nn_result.shape != ns_result.shape:
        from scipy.ndimage import zoom
        zoom_factor = (
            ns_result.shape[0] / nn_result.shape[0],
            ns_result.shape[1] / nn_result.shape[1],
            ns_result.shape[2] / nn_result.shape[2]
        )
        nn_result = zoom(nn_result, zoom_factor, order=1)
        nn_result = np.clip(nn_result, 0, cmax)

    # Compute metrics
    mae = 0.0       # Mean Absolute Error
    mse = 0.0       # Mean Squared Error
    r2 = 0.0        # R² (coefficient of determination)
    l2_error = 0.0  # Relative L2 error

    if nn_result is not None and nn_available:
        if nn_result.shape != ns_result.shape:
            from scipy.ndimage import zoom
            zoom_factor = (
                ns_result.shape[0] / nn_result.shape[0],
                ns_result.shape[1] / nn_result.shape[1],
                ns_result.shape[2] / nn_result.shape[2]
            )
            nn_result = zoom(nn_result, zoom_factor, order=1)
            nn_result = np.clip(nn_result, 0, cmax)

        diff = nn_result - ns_result

        # MAE - средняя абсолютная ошибка
        mae = np.mean(np.abs(diff))

        # MSE - среднеквадратичная ошибка
        mse = np.mean(diff ** 2)

        # R² - коэффициент детерминации (насколько хорошо модель объясняет данные)
        ss_res = np.sum(diff ** 2)
        ss_tot = np.sum((ns_result - np.mean(ns_result)) ** 2) + 1e-8
        r2 = 1 - (ss_res / ss_tot)
        r2 = max(0, r2) * 100  # в процентах, не меньше 0

        # Relative L2 error
        l2_error = np.sqrt(mse) / (np.sqrt(np.mean(ns_result ** 2)) + 1e-8)

    speedup = ns_time / (nn_time + 1e-8) if nn_time > 0 else 0.0

    # Grid coordinates matching solver cell centers
    dx = Lx / nx
    dy = Ly / ny
    x_grid = (np.linspace(0, Lx, nx, endpoint=False) + dx/2).tolist()
    y_grid = (np.linspace(0, Ly, ny, endpoint=False) + dy/2).tolist()

    print(f"[DEBUG] Grid: nx={nx}, ny={ny}, Lx={Lx}, Ly={Ly}")
    print(f"[DEBUG] x_grid: len={len(x_grid)}, range=[{x_grid[0]:.2f}, {x_grid[-1]:.2f}]")
    print(f"[DEBUG] y_grid: len={len(y_grid)}, range=[{y_grid[0]:.2f}, {y_grid[-1]:.2f}]")
    print(f"[DEBUG] NS result shape: {ns_result.shape} (expected: n_times={len(times)}, ny={ny}, nx={nx})")

    # Verify shapes match for Plotly
    if ns_result.shape[1] != len(y_grid) or ns_result.shape[2] != len(x_grid):
        print(f"[ERROR] Shape mismatch! ns_result[1]={ns_result.shape[1]} vs y_grid={len(y_grid)}, ns_result[2]={ns_result.shape[2]} vs x_grid={len(x_grid)}")

    if nn_result is None:
        nn_result = np.zeros_like(ns_result)

    # Final debug output
    print(f"[DEBUG] Final NS result: shape={ns_result.shape}, range=[{ns_result.min():.6f}, {ns_result.max():.6f}]")
    print(f"[DEBUG] Final NN result: shape={nn_result.shape}, range=[{nn_result.min():.6f}, {nn_result.max():.6f}]")

    # Check specific frames
    print(f"[DEBUG] Frame 0 (t={times[0]:.1f}s): NS max={ns_result[0].max():.6f}")
    mid_frame = len(times) // 2
    print(f"[DEBUG] Frame {mid_frame} (t={times[mid_frame]:.1f}s): NS max={ns_result[mid_frame].max():.6f}")
    print(f"[DEBUG] Frame -1 (t={times[-1]:.1f}s): NS max={ns_result[-1].max():.6f}")

    # Use actual max concentration for colorscale (at least c_in, but could be higher)
    actual_max = max(float(ns_result.max()), float(params.c_in))
    print(f"[DEBUG] Using c_max={actual_max:.4f} for colorscale (ns_max={ns_result.max():.4f}, c_in={params.c_in})")

    # Subsample frames to reduce JSON response size (every 4th frame: 201 -> 51)
    step = 4
    ns_sub = np.round(ns_result[::step], 4)
    nn_sub = np.round(nn_result[::step], 4)
    times_sub = times[::step]
    print(f"[DEBUG] Subsampled: {len(times)} -> {len(times_sub)} frames for web transfer")

    return {
        "nn": nn_sub.tolist(),
        "ns": ns_sub.tolist(),
        "times": times_sub.tolist(),
        "x_grid": [round(x, 4) for x in x_grid],
        "y_grid": [round(y, 4) for y in y_grid],
        "c_max": round(actual_max, 4),
        "nn_time": nn_time,
        "ns_time": ns_time,
        "speedup": speedup,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "l2_error": l2_error,
        "nn_available": nn_available,
        "model_warning": MODEL_WARNING,
    }


def create_gif_from_data(data: np.ndarray, x_grid: list, y_grid: list, c_max: float, fps: int = 10) -> bytes:
    """Create GIF from simulation data."""
    frames = []
    ny, nx = len(y_grid), len(x_grid)
    X, Y = np.meshgrid(np.array(x_grid), np.array(y_grid))

    for frame_idx in range(data.shape[0]):
        frame = data[frame_idx]

        fig, ax = plt.subplots(figsize=(10, 8), dpi=80)
        im = ax.contourf(
            X, Y,
            np.clip(frame, 0, c_max),
            levels=60,
            cmap='turbo',
            vmin=0,
            vmax=c_max
        )
        ax.set_xlabel('x (м)', fontsize=12)
        ax.set_ylabel('y (м)', fontsize=12)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='c', fraction=0.046)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80, facecolor='white')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)

    gif_buf = BytesIO()
    if frames:
        frames[0].save(
            gif_buf,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=1000//fps,
            loop=0,
            optimize=False
        )
    gif_buf.seek(0)
    return gif_buf.getvalue()


@app.post("/api/download/gif")
async def download_gif(params: SimulationParams):
    """Generate and download GIF from simulation."""
    dT = MODEL_META.get('dT', 2.0) if MODEL_META else 2.0
    Tmax = MODEL_META.get('Tmax', 400.0) if MODEL_META else 400.0
    cmax = MODEL_META.get('cmax', 0.635) if MODEL_META else 0.635
    Lx = MODEL_META.get('L', 60.0) if MODEL_META else 60.0
    Ly = MODEL_META.get('H', 60.0) if MODEL_META else 60.0
    nx = MODEL_META.get('nx', 100) if MODEL_META else 100
    ny = MODEL_META.get('ny', 100) if MODEL_META else 100

    Q_internal = -abs(params.Q)
    raw_params = np.array([
        params.c_in, params.w0, params.mu0, Q_internal,
        params.chi, params.c_in_times, dT
    ], dtype=np.float32)

    nn_result = None
    if MODEL is not None and MODEL_VALID:
        try:
            norm_params = normalize_params(raw_params)
            inp = torch.from_numpy(norm_params.reshape(1, -1)).to(DEVICE).float()
            with torch.no_grad():
                pred = MODEL(inp)[0].cpu().numpy()
            pred = pred.transpose(0, 2, 1)
            nn_result = pred * cmax
        except Exception as e:
            print(f"NN prediction error: {e}")

    try:
        c_in_times_arr = np.array([params.c_in_times])
        c_in_arr = np.array([params.c_in, 0.0])
        inlet_fraction = min(params.chi / Ly, 1.0)

        solver = ProppantSolver(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, T=Tmax, dT=dT,
            c_inlet=params.c_in, Q_inlet=Q_internal, g=0.0,
            mu0=params.mu0, r_particle=0.0, w0=params.w0,
            inlet_fraction=inlet_fraction,
            c_in_times=c_in_times_arr, c_in_arr=c_in_arr,
            rk_stages=3, lim_type='koren',
        )

        times, concentrations = solver.solve()
        ns_result = np.clip(concentrations, 0, cmax)

        dx = Lx / nx
        dy = Ly / ny
        x_grid = (np.linspace(0, Lx, nx, endpoint=False) + dx/2).tolist()
        y_grid = (np.linspace(0, Ly, ny, endpoint=False) + dy/2).tolist()

        result_data = nn_result if nn_result is not None else ns_result
        actual_max = max(float(result_data.max()), float(params.c_in))

        gif_data = create_gif_from_data(result_data, x_grid, y_grid, actual_max, fps=10)

        filename = f"simulation_c{params.c_in}_w{params.w0}_Q{params.Q}.gif"
        return Response(
            content=gif_data,
            media_type="image/gif",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating GIF: {str(e)}")


@app.post("/api/download/frame")
async def download_frame(params: SimulationParams, frame_idx: int = 0):
    """Generate and download current frame as PNG."""
    dT = MODEL_META.get('dT', 2.0) if MODEL_META else 2.0
    Tmax = MODEL_META.get('Tmax', 400.0) if MODEL_META else 400.0
    cmax = MODEL_META.get('cmax', 0.635) if MODEL_META else 0.635
    Lx = MODEL_META.get('L', 60.0) if MODEL_META else 60.0
    Ly = MODEL_META.get('H', 60.0) if MODEL_META else 60.0
    nx = MODEL_META.get('nx', 100) if MODEL_META else 100
    ny = MODEL_META.get('ny', 100) if MODEL_META else 100

    Q_internal = -abs(params.Q)
    raw_params = np.array([
        params.c_in, params.w0, params.mu0, Q_internal,
        params.chi, params.c_in_times, dT
    ], dtype=np.float32)

    nn_result = None
    if MODEL is not None and MODEL_VALID:
        try:
            norm_params = normalize_params(raw_params)
            inp = torch.from_numpy(norm_params.reshape(1, -1)).to(DEVICE).float()
            with torch.no_grad():
                pred = MODEL(inp)[0].cpu().numpy()
            pred = pred.transpose(0, 2, 1)
            nn_result = pred * cmax
        except Exception as e:
            print(f"NN prediction error: {e}")

    try:
        c_in_times_arr = np.array([params.c_in_times])
        c_in_arr = np.array([params.c_in, 0.0])
        inlet_fraction = min(params.chi / Ly, 1.0)

        solver = ProppantSolver(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, T=Tmax, dT=dT,
            c_inlet=params.c_in, Q_inlet=Q_internal, g=0.0,
            mu0=params.mu0, r_particle=0.0, w0=params.w0,
            inlet_fraction=inlet_fraction,
            c_in_times=c_in_times_arr, c_in_arr=c_in_arr,
            rk_stages=3, lim_type='koren',
        )

        times, concentrations = solver.solve()
        ns_result = np.clip(concentrations, 0, cmax)

        dx = Lx / nx
        dy = Ly / ny
        x_grid = (np.linspace(0, Lx, nx, endpoint=False) + dx/2).tolist()
        y_grid = (np.linspace(0, Ly, ny, endpoint=False) + dy/2).tolist()

        result_data = nn_result if nn_result is not None else ns_result
        actual_max = max(float(result_data.max()), float(params.c_in))

        frame_idx = min(max(0, frame_idx), result_data.shape[0] - 1)
        frame = result_data[frame_idx]

        X, Y = np.meshgrid(np.array(x_grid), np.array(y_grid))
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        im = ax.contourf(
            X, Y,
            np.clip(frame, 0, actual_max),
            levels=60,
            cmap='turbo',
            vmin=0,
            vmax=actual_max
        )
        ax.set_xlabel('x (м)', fontsize=12)
        ax.set_ylabel('y (м)', fontsize=12)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='c', fraction=0.046)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        buf.seek(0)
        plt.close(fig)

        filename = f"frame_{frame_idx}_c{params.c_in}_w{params.w0}.png"
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating frame: {str(e)}")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_valid": MODEL_VALID,
        "model_warning": MODEL_WARNING,
        "device": DEVICE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8050)
