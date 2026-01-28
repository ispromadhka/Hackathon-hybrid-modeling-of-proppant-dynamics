# Proppant Transport Hybrid Simulator

Neural network surrogate model (SuperB-FNO) for accelerating proppant transport simulations in hydraulic fracturing.

## Quick Start

### New Web Interface (FastAPI)

```bash
# Install dependencies
pip install -r requirements.txt

# Run new web interface
python web/server.py

# Open http://localhost:8050
```

Features:
- Light/Dark theme toggle
- Real-time NN vs NS comparison
- Synchronized timelapse with play/pause
- Interactive Plotly.js plots with hover
- Metrics: speedup, L2 error, computation times

### Legacy Dash Interface

```bash
python app.py --port 8050
```

## Pretrained Model

If you don't have a trained model, the system will use a pretrained checkpoint.

To specify a custom pretrained model, set the path in `checkpoints/best.pt` or modify `web/server.py`:
```python
CHECKPOINT_PATH = ROOT / 'checkpoints' / 'best.pt'
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

```bash
# Generate training data
python app.py --generate --samples 500 --workers 8

# Train SuperB-FNO model
python app.py --train --epochs 100 --patience 15

# Run legacy Dash app
python app.py --port 8050
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--generate` | - | Generate training data |
| `--train` | - | Train FNO model |
| `--samples` | 500 | Number of samples to generate |
| `--workers` | 1 | Parallel workers (-1 = all CPUs) |
| `--epochs` | 100 | Training epochs |
| `--patience` | 15 | Early stopping patience |
| `--port` | 8050 | Web app port |

## SuperB-FNO Architecture

Multi-scale Fourier Neural Operator with:
- Residual connections
- Spectral attention
- Parameter conditioning at bottleneck
- ~8.6M parameters

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
├── app.py                 # Legacy CLI
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
│   │   └── to_torch.py    # Data conversion
│   ├── model/
│   │   └── fno.py         # SuperB-FNO architecture
│   ├── training/
│   │   ├── dataset.py     # Data generation
│   │   └── train.py       # Training loop
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

## Goal

Replace numerical solver (~5s per simulation) with SuperB-FNO for 100-1000x speedup.
