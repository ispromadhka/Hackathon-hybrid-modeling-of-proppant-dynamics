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

# Open http://127.0.0.1:8050 or http://localhost:8050 in browser
```

## CLI Commands

```bash
# Generate training data
python app.py --generate --samples 500 --workers 8

# Train FNO model
python app.py --train

# Resume training from checkpoint
python app.py --train --resume best.pt

# Override config parameters
python app.py --train --epochs 200 --lr 0.0005 --patience 10

# Run web app (default: http://127.0.0.1:8050)
python app.py --port 8050

# For network access (accessible from other devices)
python app.py --host 0.0.0.0 --port 8050
```

## Training Configuration

Training parameters are configured in `configs/default.json` under the `training` section:

```json
{
  "training": {
    "n_epochs": 100,
    "batch_size": 8,
    "train_ratio": 0.8,
    "resume_from": null,
    "optimizer": {
      "type": "AdamW",
      "lr": 0.001,
      "weight_decay": 0.0001,
      "betas": [0.9, 0.999],
      "eps": 1e-8
    },
    "scheduler": {
      "type": "cosine_with_warmup",
      "warmup_epochs": 5,
      "T_max": null,
      "eta_min": 0.00001
    },
    "loss": {
      "mse_weight": 1.0,
      "rel_weight": 0.5,
      "temporal_weight": 0.1
    },
    "trainer": {
      "patience": 8,
      "min_delta": 0.0001,
      "grad_clip": 1.0
    }
  }
}
```

**Optimizer options:**
- `type`: `"AdamW"` or `"Adam"`
- `lr`: Learning rate
- `weight_decay`: L2 regularization
- `betas`: Adam momentum parameters
- `eps`: Numerical stability epsilon

**Scheduler options:**
- `type`: `"cosine_with_warmup"` or `"step"`
- `warmup_epochs`: Number of warmup epochs (for cosine_with_warmup)
- `T_max`: Maximum iterations for cosine annealing (null = auto)
- `eta_min`: Minimum learning rate
- `step_size` and `gamma`: For step scheduler

**Loss weights:**
- `mse_weight`: Mean squared error weight
- `rel_weight`: Relative L2 error weight
- `temporal_weight`: Temporal consistency weight

**Trainer settings:**
- `patience`: Early stopping patience (epochs without improvement)
- `min_delta`: Minimum improvement threshold
- `grad_clip`: Gradient clipping norm

### Resuming Training

To resume training from a checkpoint, set `resume_from` in config or use CLI. **All checkpoint paths are relative to the `checkpoints/` directory:**

```bash
# Resume from last checkpoint (looks in checkpoints/last.pt)
python app.py --train --resume last.pt

# Resume from best checkpoint (looks in checkpoints/best.pt)
python app.py --train --resume best.pt

# You can also use subdirectories (looks in checkpoints/models/best.pt)
python app.py --train --resume models/best.pt
```

**In config file (`configs/default.json`):**
```json
{
  "training": {
    "resume_from": "best.pt"  // Relative to checkpoints/ directory
  }
}
```

Checkpoints are saved in `checkpoints/` directory:
- `checkpoints/last.pt` — Latest checkpoint
- `checkpoints/best.pt` — Best validation loss checkpoint

**Note:** If you provide an absolute path, it will be used as-is. Otherwise, the path is resolved relative to the `checkpoints/` directory.

When resuming, the trainer restores:
- Model weights
- Optimizer state
- Scheduler state
- Training epoch
- Best validation metrics

## Data Generation Modes

The dataset generation supports different sampling strategies configured in `configs/default.json`:

```json
{
  "dataset_generation": {
    "sampling": "random",  // or "lhs"
    "seed": 42
  }
}
```

**Sampling modes:**
- `"random"` — Randomly selects parameter combinations from discrete grids, avoiding duplicates with existing simulations
- `"lhs"` — Latin Hypercube Sampling for uniform parameter space coverage

Both modes automatically:
- Check for uniqueness against existing parameter combinations
- Validate parameters for physical correctness (positive values, valid ranges, etc.)
- Skip already generated simulations

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--generate` | - | Generate training data |
| `--train` | - | Train FNO model |
| `--samples` | 500 | Number of samples to generate |
| `--workers` | 1 | Parallel workers (-1 = all CPUs) |
| `--config` | `configs/default.json` | Config file path |
| `--epochs` | - | Training epochs (overrides config) |
| `--lr` | - | Learning rate (overrides config) |
| `--patience` | - | Early stopping patience (overrides config) |
| `--resume` | - | Resume from checkpoint (e.g., "last.pt") |
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

## Git Workflow

### Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd Hackathon-hybrid-modeling-of-proppant-dynamics

# Check current branch
git branch

# Switch to main branch (if not already on it)
git checkout main
```

### Making Changes and Pushing to Main

```bash
# 1. Check status of your changes
git status

# 2. Add files to staging
git add .
# Or add specific files:
git add <file1> <file2>

# 3. Commit your changes
git commit -m "Description of your changes"

# 4. Push to main branch
git push origin main
```

### Working with Branches (Recommended)

For better collaboration, consider working on feature branches:

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Make your changes, then commit
git add .
git commit -m "Add new feature"

# Push the branch to remote
git push origin feature/your-feature-name

# Merge to main (after review/approval)
git checkout main
git merge feature/your-feature-name
git push origin main
```

### Viewing History

```bash
# View commit history
git log

# Compact one-line view
git log --oneline

# With graph visualization
git log --oneline --graph --all

# View changes in a file
git log -p <filename>
```

### Common Git Commands

```bash
# Pull latest changes from remote
git pull origin main

# View differences
git diff

# View staged changes
git diff --staged

# Undo changes to a file (before staging)
git checkout -- <filename>

# Unstage a file
git reset HEAD <filename>

# View remote repositories
git remote -v
```

### Troubleshooting

If you encounter issues pushing to main:

```bash
# If main branch is protected, you may need to:
# 1. Create a pull request instead
# 2. Or request branch protection to be adjusted

# If you need to force push (use with caution!)
git push origin main --force
# ⚠️ Warning: Only use --force if you're sure no one else is working on main
```

## Goal

Replace numerical solver (~5s per simulation) with FNO for 100-1000x speedup.
