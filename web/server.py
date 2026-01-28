"""
FastAPI backend for Proppant Transport Simulator.
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
CONFIG_PATH = ROOT / 'configs' / 'default.json'
DATA_META_PATH = ROOT / 'data' / 'processed' / 'metadata.json'

# Global model
MODEL = None
MODEL_META = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app = FastAPI(title="Proppant Transport Simulator")

# Mount static files
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


class SimulationParams(BaseModel):
    c_in: float = 0.3
    w0: float = 0.005
    mu0: float = 0.001
    Q: float = 0.05  # Positive in API, negated internally
    chi: float = 5.0
    c_in_times: float = 50.0


def load_model():
    """Load the trained model."""
    global MODEL, MODEL_META

    if not CHECKPOINT_PATH.exists():
        print(f"Warning: No checkpoint found at {CHECKPOINT_PATH}")
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

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Detect model dimensions
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

    if 'project.5.weight' in state_dict:
        n_times = state_dict['project.5.weight'].shape[0]
    elif 'project.2.weight' in state_dict:
        n_times = state_dict['project.2.weight'].shape[0]
    else:
        n_times = MODEL_META.get('n_times', 201)

    MODEL = create_model(nx=nx, ny=ny, n_times=n_times, n_params=n_params, device=DEVICE)
    MODEL.load_state_dict(state_dict, strict=False)
    MODEL.eval()

    print(f"Model loaded: nx={nx}, ny={ny}, n_times={n_times}, n_params={n_params}")
    return True


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

    # Negate Q internally (user inputs positive, solver expects negative)
    Q_internal = -abs(params.Q)

    # Build raw parameter array
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

    if MODEL is not None:
        try:
            norm_params = normalize_params(raw_params)
            inp = torch.from_numpy(norm_params.reshape(1, -1)).to(DEVICE).float()

            t0 = time.perf_counter()
            with torch.no_grad():
                pred = MODEL(inp)[0].cpu().numpy()  # (n_times, nx, ny)
            nn_time = time.perf_counter() - t0

            # Transpose to (n_times, ny, nx) for consistency with solver
            pred = pred.transpose(0, 2, 1)
            # Denormalize from [0,1] to [0, cmax]
            nn_result = pred * cmax

        except Exception as e:
            print(f"NN prediction error: {e}")
            nn_result = None

    # ===== Numerical Solver =====
    ns_result = None
    ns_time = 0.0
    times = None

    try:
        # Create injection pattern
        c_in_times_arr = np.array([params.c_in_times])
        c_in_arr = np.array([params.c_in, 0.0])

        solver = ProppantSolver(
            nx=nx, ny=ny,
            Lx=Lx, Ly=Ly,
            T=Tmax, dT=dT,
            c_inlet=params.c_in,
            Q_inlet=Q_internal,
            g=9.81,
            mu0=params.mu0,
            r_particle=0.0002,
            inlet_fraction=0.5,
            c_in_times=c_in_times_arr,
            c_in_arr=c_in_arr,
            rk_stages=3,
            lim_type='koren',
        )

        t0 = time.perf_counter()
        times, concentrations = solver.solve()
        ns_time = time.perf_counter() - t0

        ns_result = np.clip(concentrations, 0, cmax)

    except Exception as e:
        print(f"NS solver error: {e}")
        raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")

    if ns_result is None:
        raise HTTPException(status_code=500, detail="Solver failed to produce results")

    # Match NN and NS frame counts
    n_frames = ns_result.shape[0]
    if nn_result is not None:
        nn_frames = nn_result.shape[0]
        if nn_frames != n_frames:
            # Interpolate NN to match NS frames
            from scipy.ndimage import zoom
            zoom_factor = (n_frames / nn_frames, 1, 1)
            nn_result = zoom(nn_result, zoom_factor, order=1)
            nn_result = np.clip(nn_result, 0, cmax)

    # Compute L2 error if both available
    l2_error = 0.0
    if nn_result is not None:
        diff = nn_result - ns_result
        l2_error = np.sqrt(np.mean(diff ** 2)) / (np.sqrt(np.mean(ns_result ** 2)) + 1e-8)

    # Compute speedup
    speedup = ns_time / (nn_time + 1e-8) if nn_time > 0 else 0.0

    # Create coordinate grids
    x_grid = np.linspace(0, Lx, nx).tolist()
    y_grid = np.linspace(0, Ly, ny).tolist()

    # Use placeholder if NN not available
    if nn_result is None:
        nn_result = np.zeros_like(ns_result)

    return {
        "nn": nn_result.tolist(),
        "ns": ns_result.tolist(),
        "times": times.tolist(),
        "x_grid": x_grid,
        "y_grid": y_grid,
        "c_max": float(params.c_in),
        "nn_time": nn_time,
        "ns_time": ns_time,
        "speedup": speedup,
        "l2_error": l2_error
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": DEVICE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
