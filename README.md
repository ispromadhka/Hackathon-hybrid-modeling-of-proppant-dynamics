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

# Train FNO model
python app.py --train --epochs 100

# Run web app
python app.py --port 8050
```

## Physics

Two-phase proppant transport model:

```
∂(cw)/∂t + ∇·(cwVₚ) = 0           — mass conservation
Vf = -w²/(12μ(c)) · (∇P - ρ(c)g)  — Darcy flow
Vₚ = Vf + Vslip(c)                 — proppant velocity
```

- `c` — proppant concentration
- `w` — fracture width
- `μ(c)` — concentration-dependent viscosity (Krieger-Dougherty)
- `Vslip` — settling velocity (Richardson-Zaki)

## Project Structure

```
├── app.py                 # Main entry point
├── src/
│   ├── solver/            # Numerical solvers
│   │   └── proppant_transport.py
│   ├── model/             # FNO neural network
│   │   └── fno.py
│   ├── training/          # Training pipeline
│   └── visualization/     # Web interface
│       └── app.py
├── data/                  # Training data
├── checkpoints/           # Model weights
└── configs/               # Configuration files
```

## Web Interface

- Adjust initial conditions (Gaussian, Step, Layer)
- Change physics parameters (gravity, viscosity, fracture width)
- Real-time visualization with animation controls

## Goal

Replace slow numerical solver with FNO for ~100x speedup while maintaining accuracy.
