# Proppant Transport Hybrid Simulator

Neural network surrogate model (SuperB-FNO) for accelerating proppant transport simulations in hydraulic fracturing.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run web interface
python app.py --port 8050

# Open http://localhost:8050
```

### Features
- Light/Dark theme toggle (button top-right, default: light)
- Keyboard input for all parameters
- Synchronized timelapse slider (drag manually or use Play/Pause)
- Interactive Plotly.js heatmaps with hover tooltips
- Colorbar on each plot
- Metrics: NN time, NS time, speedup, L2 error

### Legacy Dash Interface

```bash
python app.py --legacy --port 8050
```

## Docker

```bash
# Build and run web interface
docker-compose up app

# Generate training data (CPU parallelized)
docker-compose run generate

# Train model (GPU)
docker-compose run train
```

## CLI Commands

### Data Generation (CPU Parallelized)

```bash
# Auto-detect optimal workers (75% of CPUs)
python app.py --generate --samples 1000 --workers -1

# Specify exact number of workers
python app.py --generate --samples 1000 --workers 8

# Sequential processing
python app.py --generate --samples 100 --workers 0
```

### Training (GPU Accelerated)

```bash
# Full parallelization (AMP + Multi-GPU)
python app.py --train --epochs 100 --patience 20

# Disable mixed precision
python app.py --train --epochs 100 --no-amp

# Single GPU mode
python app.py --train --epochs 100 --no-multi-gpu

# Gradient accumulation for larger effective batch size
python app.py --train --epochs 100 --grad-accum 4

# Custom data loading workers
python app.py --train --epochs 100 --num-workers 8

# Disable physics-informed loss
python app.py --train --epochs 100 --no-physics-loss
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--generate` | - | Generate training data |
| `--train` | - | Train FNO model |
| `--samples` | 500 | Number of samples to generate |
| `--workers` | 1 | CPU workers for generation (-1 = auto) |
| `--epochs` | 100 | Training epochs |
| `--patience` | 15 | Early stopping patience |
| `--lr` | 1e-3 | Learning rate |
| `--no-amp` | - | Disable mixed precision training |
| `--no-multi-gpu` | - | Disable multi-GPU DataParallel |
| `--num-workers` | 4 | Data loading workers |
| `--grad-accum` | 1 | Gradient accumulation steps |
| `--physics-loss` | true | Use physics-informed losses |
| `--no-physics-loss` | - | Disable physics-informed losses |
| `--port` | 8050 | Web app port |

## SuperB-FNO Architecture

Enhanced multi-scale Fourier Neural Operator with:

- **High-Frequency Scaling (HFS)** — mitigates spectral bias, improves fine detail learning
- **Multi-band Spectral Attention** — 4-band frequency weighting with learnable boundaries
- **SpecBoost-style Residual Learning** — separate low/high frequency paths
- **Boundary Masks** — enforces mass conservation at domain boundaries
- **Multi-scale Fusion** — learnable scale mixing with modes [(32,32), (24,24), (16,16), (12,12), (8,8), (6,6)]
- **Parameter Conditioning** — injection at multiple network depths
- **~24M parameters**

### Physics-Informed Losses

- **Spectral Loss** — frequency-domain loss with 2x weight on high frequencies
- **Mass Conservation Loss** — penalizes mass violations and non-smooth mass changes
- **Boundary Loss** — ensures correct boundary conditions (2x weight on inlet)
- **Temporal Consistency Loss** — smooth temporal evolution

**Input (7 parameters):**
| Parameter | Description | Range |
|-----------|-------------|-------|
| `c_in` | Inlet concentration | [0.1, 0.5] |
| `w0` | Fracture aperture | [0.001, 0.01] m |
| `mu0` | Base viscosity | [0.0001, 0.01] Pa·s |
| `Q` | Flow rate | [0.01, 0.1] m²/s |
| `chi` | Chi parameter | [1.0, 10.0] |
| `c_in_times` | Injection time | [10, 200] s |
| `dT` | Time step | [1.0, 5.0] s |

> **Note**: Q is entered as positive in the UI but internally negated for the solver.

**Output:**
- Concentration field c(x,y,t) over 201 time steps
- Grid: 100×100 (Lx=60m, Ly=60m)

## Parallelization

### CPU (Data Generation)

- Automatic worker count detection (75% of available CPUs)
- NumPy threading disabled per worker to avoid oversubscription
- Timeout protection (5 min per simulation)
- Incremental result saving (every 5%)
- Progress tracking with ETA

### GPU (Training)

- **Mixed Precision (AMP)** — 2-3x speedup on CUDA GPUs
- **DataParallel** — automatic multi-GPU distribution
- **Gradient Accumulation** — larger effective batch size without memory increase
- **Optimized Data Loading:**
  - `pin_memory` for fast GPU transfer
  - `persistent_workers` for fast epoch transitions
  - `prefetch_factor=2` for data prefetching
- **Auto batch size** based on GPU memory

## Physics Model

Two-phase proppant transport:

```
∂(cw)/∂t + ∇·(cwVₚ) = 0              — mass conservation
Vf = -w²/(12μ(c)) · (∇P - ρ(c)g)     — Darcy's law
Vₚ = Vf + Vslip(c)                    — proppant velocity
```

**Closure models:**
- `μ(c) = μ₀(1 - c/c_max)^(-β)` — Krieger-Dougherty viscosity
- `Vslip = V_stokes·(1 - c/c_max)^n` — Richardson-Zaki settling

**Numerical methods:**
- Pressure: Sparse direct solver
- Transport: TVD (Koren limiter) + RK3
- CFL-adaptive time stepping

## Project Structure

```
├── app.py                 # Main CLI entry point
├── web/
│   ├── server.py          # FastAPI backend
│   ├── templates/
│   │   └── index.html     # Main page
│   └── static/
│       ├── css/style.css  # Styles with theme support
│       └── js/app.js      # Frontend logic
├── src/
│   ├── solver/
│   │   ├── CPU_solver/    # Physics solver
│   │   ├── generation.py  # Parallel data generation
│   │   └── to_torch.py    # Data conversion
│   ├── model/
│   │   └── fno.py         # SuperB-FNO architecture
│   ├── training/
│   │   ├── dataset.py     # Dataset & dataloaders
│   │   └── train.py       # Training with AMP/Multi-GPU
│   └── visualization/
│       └── app.py         # Legacy Dash UI
├── checkpoints/           # Model weights (best.pt)
├── configs/               # Generation configs
└── data/processed/        # Training data
```

## Training Metrics

The training script shows:
- **MAE** — Mean Absolute Error
- **Acc%** — Predictions within 5% tolerance
- **R²%** — Coefficient of determination
- **Mass%** — Mass conservation error

## Pretrained Model

If you don't have a trained model, the system will use a pretrained checkpoint.

To specify a custom pretrained model, set the path in `checkpoints/best.pt` or modify `web/server.py`:
```python
CHECKPOINT_PATH = ROOT / 'checkpoints' / 'best.pt'
```

## Goal

Replace numerical solver (~5s per simulation) with SuperB-FNO for 100-1000x speedup while maintaining physical consistency through physics-informed training.
