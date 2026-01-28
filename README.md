# Proppant Transport Hybrid Simulator

Neural network surrogate model (FNO) for accelerating proppant transport simulations in hydraulic fracturing.

## Quick Start with Docker

```bash
# Build and run web interface
docker-compose up app

# Open http://localhost:8050 in browser
```

### Generate Training Data (CPU parallelized)

```bash
docker-compose run generate
```

This runs 8 parallel workers to generate 1000 samples. Edit `docker-compose.yml` to change:
- `--samples 1000` — number of simulations
- `--workers 8` — parallel CPU workers

### Train FNO Model (GPU)

```bash
docker-compose run train
```

Requires NVIDIA GPU with Docker GPU support.

## Local Installation

```bash
pip install -r requirements.txt
python app.py --port 8050
```

## CLI Commands

```bash
# Generate training data
python app.py --generate --samples 500 --workers 8

# Train FNO model
python app.py --train --epochs 100 --patience 15

# Run web app
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

## FNO Architecture

**Input (9 parameters):**
- `c_inlet` — inlet concentration [0.15, 0.40]
- `Q_inlet` — flow rate [0.02, 0.10] m²/s
- `g` — gravity [0.0, 9.81] m/s²
- `mu0` — base viscosity [0.001, 0.010] Pa·s
- `r_particle` — particle radius [0.0001, 0.0003] m
- `inlet_fraction` — inlet height fraction [0.3, 0.7]
- `rk_stages` — RK3 time integration (fixed)
- `lim_type` — Koren limiter (fixed)
- `injection_mode` — injection pattern [0, 1, 2]

**Output:**
- Concentration field c(x,y,t) over 100 time steps
- Grid: 120×60 (Lx=60m, Ly=30m)

**Architecture:**
- 4 Fourier layers
- Width: 48 channels
- Fourier modes: (12, 8)
- ~10M parameters

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
├── app.py                 # Main CLI
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── solver/
│   │   ├── CPU_solver/    # Physics solver
│   │   └── to_torch.py    # Data conversion
│   ├── model/
│   │   └── fno.py         # FNO architecture
│   ├── training/
│   │   ├── dataset.py     # Data generation
│   │   └── train.py       # Training loop
│   └── visualization/
│       └── app.py         # Dash web UI
├── data/                  # Training data
├── checkpoints/           # Model weights
└── configs/               # Generation configs
```

## Data Pipeline

1. **Generate**: `simulation_timeseries/*_series.npz`
2. **Convert**: `torch_data/data.pt`
3. **Process**: `data/processed/sample_*.npz`

## Goal

Replace numerical solver (~5s per simulation) with FNO for 100-1000x speedup.
