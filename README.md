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
# Generate training data (500 samples)
python app.py --generate --samples 500

# Train FNO model (100 epochs)
python app.py --train --epochs 100

# Run web app on custom port
python app.py --port 8050
```

## Physics

Two-phase proppant transport in a fracture:

```
∂c/∂t + ∇·(c·Vₚ) = 0              — mass conservation
Vf = Poiseuille profile            — slot flow (parabolic velocity)
Vₚ = Vf + Vslip(c)                 — proppant velocity
Vslip = -V_stokes·(1-c/c_max)^n    — Richardson-Zaki settling
```

**Features:**
- Continuous proppant injection at inlet (x=0)
- Poiseuille flow profile: u(y) = U_max·4y(1-y)
- Gravity settling with hindered settling correction
- Nolte viscosity model for slurry
- Impermeable walls (top/bottom)

**Parameters:**
- `c` — proppant volume concentration
- `μ(c)` — concentration-dependent viscosity (Nolte model)
- `Vslip` — settling velocity (Richardson-Zaki)
- `d_p` — particle diameter (200-800 μm)

## Project Structure

```
├── app.py                 # Main entry point (CLI)
├── src/
│   ├── solver/            # Numerical solvers
│   │   └── proppant_transport.py
│   ├── model/             # FNO neural network
│   │   └── fno.py
│   ├── training/          # Training pipeline
│   │   ├── dataset.py     # Data generation
│   │   └── train.py       # Training loop
│   └── visualization/     # Web interface
│       └── app.py
├── data/                  # Training data
├── checkpoints/           # Model weights
└── configs/               # Configuration files
```

## Web Interface

Interactive simulator with adjustable parameters:

- **Inlet concentration** (c₀) — proppant concentration at inlet
- **Flow velocity** (U_max) — maximum Poiseuille velocity
- **Gravity** (g) — gravitational acceleration
- **Viscosity** (μ) — fluid viscosity
- **Particle diameter** (d_p) — affects settling rate

Real-time visualization with Play/Pause animation controls.

## FNO Model

Fourier Neural Operator architecture:
- Input: Physics parameters [c_inlet, U_max, g, μ, d_p]
- Output: Full trajectory c(x,y,t) for all time steps
- ~1.8M parameters
- Trained with relative L2 loss

## Training

```bash
# Generate dataset with varied physics parameters
python app.py --generate --samples 500

# Train model (GPU recommended)
python app.py --train --epochs 100
```

Checkpoints saved to `checkpoints/best.pt`.

## Goal

Replace slow numerical solver (~100ms per simulation) with FNO for ~10-100x speedup while maintaining accuracy.
