"""
FastAPI backend for Proppant Transport Simulator.
Supports both old (complex weights) and new model formats.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.fno import create_model
from src.solver.solver_wrapper import ProppantSolver

# Paths
ROOT = Path(__file__).parent.parent
WEB_DIR = Path(__file__).parent
CHECKPOINT_PATH = ROOT / 'checkpoints' / 'best.pt'
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

    if not CHECKPOINT_PATH.exists():
        MODEL_WARNING = f"Файл весов НС не найден: {CHECKPOINT_PATH}. Требуется обучение модели."
        print(f"Предупреждение: {MODEL_WARNING}")
        return False

    # Load metadata
    if DATA_META_PATH.exists():
        with open(DATA_META_PATH) as f:
            MODEL_META = json.load(f)
    else:
        MODEL_META = {
            'nx': 100,
            'ny': 100,
            'n_times': 201,
            'param_names': ['c_in', 'w0', 'mu0', 'Q', 'chi', 'c_in_times', 'dT'],
            'param_min': [0.1, 0.001, 0.0001, -0.1, 1.0, 10.0, 1.0],
            'param_max': [0.5, 0.01, 0.01, -0.01, 10.0, 200.0, 5.0],
            'cmax': 0.635,
            'L': 60.0,
            'H': 60.0,
            'Tmax': 400.0,
            'dT': 2.0
        }

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
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

    # Detect n_times from project layer
    n_times = None
    for key in ['project.6.weight', 'project.6.bias', 'project.4.weight', 'project.2.weight']:
        if key in state_dict:
            shape = state_dict[key].shape
            if 'bias' in key:
                n_times = shape[0]
            elif 'weight' in key and len(shape) == 4:
                n_times = shape[0]
            if n_times:
                break

    if n_times is None:
        n_times = MODEL_META.get('n_times', 201)

    print(f"Создание модели НС: nx={nx}, ny={ny}, n_times={n_times}, n_params={n_params}")
    MODEL = create_model(nx=nx, ny=ny, n_times=n_times, n_params=n_params, device=DEVICE)

    # Load state dict directly (old model uses complex weights which is fine)
    try:
        missing, unexpected = MODEL.load_state_dict(state_dict, strict=False)

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

    # Match frame counts
    n_frames = ns_result.shape[0]
    if nn_result is not None:
        nn_frames = nn_result.shape[0]
        if nn_frames != n_frames:
            from scipy.ndimage import zoom
            zoom_factor = (n_frames / nn_frames, 1, 1)
            nn_result = zoom(nn_result, zoom_factor, order=1)
            nn_result = np.clip(nn_result, 0, cmax)

    # Compute L2 error
    l2_error = 0.0
    if nn_result is not None:
        diff = nn_result - ns_result
        l2_error = np.sqrt(np.mean(diff ** 2)) / (np.sqrt(np.mean(ns_result ** 2)) + 1e-8)

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

    return {
        "nn": nn_result.tolist(),
        "ns": ns_result.tolist(),
        "times": times.tolist(),
        "x_grid": x_grid,
        "y_grid": y_grid,
        "c_max": actual_max,
        "nn_time": nn_time,
        "ns_time": ns_time,
        "speedup": speedup,
        "l2_error": l2_error,
        "nn_available": nn_available,
        "model_warning": MODEL_WARNING,
    }


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
    uvicorn.run(app, host="0.0.0.0", port=8050)
