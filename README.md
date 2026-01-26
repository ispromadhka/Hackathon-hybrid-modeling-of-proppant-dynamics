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
# Generate training data (100 samples)
python app.py --generate --samples 100

# Train FNO model (100 epochs)
python app.py --train --epochs 100

# Run web app on custom port
python app.py --port 8050
```

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
- Transport: TVD/WENO5 reconstruction + RK3 time integration
- CFL-adaptive time stepping

## Parameters

- `c` — proppant volume concentration [0, 0.635]
- `w` — fracture aperture [m]
- `μ₀` — base fluid viscosity [Pa·s]
- `β` — viscosity exponent (2.5)
- `ρ_f`, `ρ_p` — fluid/proppant densities [kg/m³]
- `g` — gravity [m/s²]
- `r` — particle radius [m]

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
└── configs/               # Configuration files
```

## Web Interface

Interactive simulator:
- **c₀** — inlet proppant concentration
- **Q** — flow rate [m²/s]
- **g** — gravity [m/s²]
- **μ₀** — fluid viscosity [mPa·s]
- **r** — particle radius [μm]
- **T** — simulation time [s]

## FNO Model

Fourier Neural Operator:
- Input: Physics parameters [c_inlet, Q, g, μ, r]
- Output: Full trajectory c(x,y,t)
- ~1.8M parameters
- Relative L2 loss

## Training

```bash
# Generate dataset
python app.py --generate --samples 500

# Train model (GPU recommended)
python app.py --train --epochs 100
```

## Goal

Replace numerical solver (~5-10s per simulation) with FNO for 100-1000x speedup.
