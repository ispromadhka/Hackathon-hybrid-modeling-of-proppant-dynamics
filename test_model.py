#!/usr/bin/env python3
"""Test script to diagnose FNO model behavior."""

import torch
import numpy as np
from pathlib import Path
import json

# Add parent to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.model.fno_v2 import create_enhanced_model


def test_model():
    checkpoint_path = Path('checkpoints/best_v2.pt')
    data_path = Path('data/processed/metadata.json')

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # Load metadata
    with open(data_path) as f:
        metadata = json.load(f)
    print(f"Metadata: nx={metadata['nx']}, ny={metadata['ny']}")

    # Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    print(f"\nCheckpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

    # Check some key weights
    print(f"\nState dict keys (first 10): {list(state_dict.keys())[:10]}")
    print(f"Total keys: {len(state_dict)}")

    # Check combination_weights for SpecBoost
    if 'combination_weights' in state_dict:
        cw = state_dict['combination_weights']
        print(f"\nSpecBoost combination_weights: {cw}")
        print(f"  After softmax: {torch.softmax(cw, dim=0)}")

    # Check if model is SpecBoost
    is_specboost = any('base_model.' in k for k in state_dict.keys())
    print(f"\nIs SpecBoost: {is_specboost}")

    # Detect architecture
    prefix = 'base_model.' if is_specboost else ''
    lift_key = f'{prefix}lift.0.weight'
    if lift_key in state_dict:
        width = state_dict[lift_key].shape[0]
        in_channels = state_dict[lift_key].shape[1]
        n_params = in_channels - 2
        print(f"Width: {width}, n_params: {n_params}")

    # Detect model size
    if width >= 192:
        model_size = 'xlarge'
    elif width >= 128:
        model_size = 'large'
    elif width >= 96:
        model_size = 'medium'
    else:
        model_size = 'small'
    print(f"Model size: {model_size}")

    # Create model
    print(f"\nCreating model...")
    model = create_enhanced_model(
        nx=metadata['nx'],
        ny=metadata['ny'],
        n_times=26,
        n_params=9,
        device=device,
        model_size=model_size,
        use_specboost=is_specboost
    )

    # Load weights
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Model loaded successfully")

    # Test with various inputs
    print(f"\n{'='*60}")
    print("TESTING MODEL OUTPUT")
    print(f"{'='*60}")

    # Test params similar to what app uses
    test_cases = [
        # [c_inlet/0.5, Q/0.1, g/12, mu/0.01, r/0.0005, inlet_frac, rk, lim, inj_mode]
        ("Default params", [0.7, 0.5, 0.0, 0.1, 0.4, 0.5, 0.0, 0.0, 0.5]),
        ("High concentration", [1.0, 0.5, 0.0, 0.1, 0.4, 0.5, 0.0, 0.0, 0.5]),
        ("With gravity", [0.7, 0.5, 0.8, 0.1, 0.4, 0.5, 0.0, 0.0, 0.5]),
        ("All zeros", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("All ones", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ("Random", list(np.random.rand(9))),
    ]

    for name, params_list in test_cases:
        params = torch.tensor([params_list], dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model(params)

        out_np = output[0].cpu().numpy()
        print(f"\n{name}:")
        print(f"  Input: {params_list[:5]}...")
        print(f"  Output shape: {out_np.shape}")
        print(f"  Range: [{out_np.min():.4f}, {out_np.max():.4f}]")
        print(f"  Mean: {out_np.mean():.4f}, Std: {out_np.std():.4f}")
        print(f"  Timestep means: t0={out_np[0].mean():.4f}, t_mid={out_np[len(out_np)//2].mean():.4f}, t_last={out_np[-1].mean():.4f}")

        # Check if output varies spatially
        spatial_var = out_np[0].var()
        print(f"  Spatial variance (t0): {spatial_var:.6f}")

    # Test with real data sample
    print(f"\n{'='*60}")
    print("TESTING WITH REAL DATA SAMPLE")
    print(f"{'='*60}")

    sample_path = Path('data/processed/sample_00000.npz')
    if sample_path.exists():
        with np.load(sample_path) as data:
            real_params = torch.tensor([data['params']], dtype=torch.float32, device=device)
            real_target = data['concentrations']

        print(f"Real params: {real_params[0].cpu().numpy()}")
        print(f"Real target shape: {real_target.shape}")
        print(f"Real target range: [{real_target.min():.4f}, {real_target.max():.4f}]")
        print(f"Real target mean: {real_target.mean():.4f}")

        with torch.no_grad():
            pred = model(real_params)

        pred_np = pred[0].cpu().numpy()
        # Transpose to match target: (n_times, nx, ny) -> (n_times, ny, nx)
        # Scale from [0, 1] (Sigmoid) to [0, 0.635] (concentration)
        pred_np = pred_np.transpose(0, 2, 1) * 0.635

        print(f"\nPrediction shape: {pred_np.shape}")
        print(f"Prediction range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
        print(f"Prediction mean: {pred_np.mean():.4f}")

        # Compare
        n_times = min(pred_np.shape[0], real_target.shape[0])
        diff = np.abs(pred_np[:n_times] - real_target[:n_times])
        print(f"\nDifference (first {n_times} timesteps):")
        print(f"  Mean absolute error: {diff.mean():.4f}")
        print(f"  Max error: {diff.max():.4f}")

        # Check if prediction captures the pattern
        target_nonzero = (real_target > 0.01).sum()
        pred_nonzero = (pred_np > 0.01).sum()
        print(f"\n  Target nonzero cells: {target_nonzero}")
        print(f"  Pred nonzero cells: {pred_nonzero}")


if __name__ == '__main__':
    test_model()
