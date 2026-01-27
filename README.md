# Proppant Transport Hybrid Simulator

Neural network surrogate model for accelerating proppant transport simulations in hydraulic fracturing.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run web interface
python app.py

# Open http://localhost:8050 in browser
```

## Commands

```bash
# Generate training data
python app.py --generate --samples 500 --workers -1

# Train FNO model
python app.py --train --epochs 100 --lr 1e-3 --patience 15

# Run web app on custom port
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
| `--lr` | 1e-3 | Learning rate |
| `--patience` | 15 | Early stopping patience |
| `--port` | 8050 | Web app port |
| `--debug` | False | Debug mode |

## Physics

Full two-phase proppant transport model:

```
∂(cw)/∂t + ∇·(cwVₚ) = 0              — mass conservation
Vf = -w²/(12μ(c)) · (∇P - ρ(c)g)     — Darcy's law
Vₚ = Vf + Vslip(c)                    — proppant velocity
∇·(Vf + c·Vslip) = 0                  — incompressibility
```

**Closure models:**
- `μ(c) = μ₀(1 - c/c_max)^(-β)` — Krieger-Dougherty viscosity
- `ρ(c) = ρ_f(1-c) + ρ_p·c` — mixture density
- `Vslip = V_stokes·(1 - c/c_max)^n` — Richardson-Zaki hindered settling

**Numerical methods:**
- Pressure: Sparse direct solver / Conjugate Gradient
- Transport: TVD/WENO5 reconstruction + RK2/RK3 time integration
- CFL-adaptive time stepping

## Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `c` | Proppant volume concentration | [0, 0.635] |
| `w` | Fracture aperture | [m] |
| `μ₀` | Base fluid viscosity | [Pa·s] |
| `β` | Viscosity exponent | 2.5 |
| `ρ_f`, `ρ_p` | Fluid/proppant densities | [kg/m³] |
| `g` | Gravity | [m/s²] |
| `r` | Particle radius | [m] |

## Project Structure

```
├── app.py                 # Main entry point (CLI)
├── src/
│   ├── solver/
│   │   ├── CPU_solver/    # Full physics solver
│   │   │   ├── SystemSolverCPU.py
│   │   │   ├── PoissonCPU.py
│   │   │   ├── TransportCPU.py
│   │   │   ├── TVD_CPU.py
│   │   │   └── WENO5CPU.py
│   │   └── solver_wrapper.py
│   ├── model/             # FNO neural network
│   │   └── fno.py
│   ├── training/          # Training pipeline
│   │   ├── dataset.py
│   │   └── train.py
│   └── visualization/     # Web interface
│       └── app.py
├── data/                  # Training data
├── checkpoints/           # Model weights
└── requirements.txt
```

## Web Interface

Side-by-side comparison of Neural Network vs Numerical Solver:

**Parameters:**
- `c₀` — inlet proppant concentration
- `Q` — flow rate [m²/s]
- `g` — gravity [m/s²]
- `μ₀` — fluid viscosity [mPa·s]
- `r` — particle radius [μm]
- `Injection Duration` — proppant injection time [s]
- `T` — total simulation time [s]

**Outputs:**
- NN vs NS visualization with animation
- Computation time comparison
- Relative L2 error over time
- Speedup factor

## FNO Model

Fourier Neural Operator architecture:

- **Input**: Physics parameters (9 values)
  - `[c_inlet, Q_inlet, g, μ₀, r_particle, inlet_fraction, rk_stages, lim_type, injection_mode]`
- **Output**: Full trajectory c(x,y,t)
- **Architecture**: 4 Fourier layers, width=48, modes=(12,8)
- **Parameters**: ~10M trainable

## Training

**Loss Function**: Relative L2 Loss
```
Loss = ||pred - target||₂ / ||target||₂
```

This is better than MSE for PDEs because it's scale-invariant.

**Features:**
- Warmup + Cosine Annealing LR scheduler
- Early stopping with configurable patience
- Gradient clipping (max_norm=1.0)
- AdamW optimizer with weight decay

**Example:**
```bash
# Generate dataset with parallel workers
python app.py --generate --samples 500 --workers -1

# Train with early stopping
python app.py --train --epochs 100 --lr 1e-3 --patience 15
```

**Training output:**
```
Epoch   1 | Train: 2.34e-01 | Val: 1.89e-01 | Quality: 81.1% | LR: 2.00e-04 | Time: 12.3s [BEST]
Epoch   2 | Train: 1.56e-01 | Val: 1.45e-01 | Quality: 85.5% | LR: 4.00e-04 | Time: 11.8s [BEST]
...
```

## Data Generation

Uses Latin Hypercube Sampling for uniform parameter coverage:

- **Injection modes**: continuous, single_pulse, multi_pulse
- **Limiter types**: koren, superbee, minmod
- **RK stages**: 2, 3

## Goal

Replace numerical solver (~5s per simulation) with FNO for 100-1000x speedup while maintaining accuracy.

## Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed documentation:
- Why FNO instead of PINN
- Neural network architecture explanation
- Training methodology and loss functions
- Physics solver details
