#!/usr/bin/env python3
"""
Main entry point for Proppant Dynamics Hybrid Simulator.

Usage:
    python app.py                          # Run web app
    python app.py --generate --samples 500 # Generate training data
    python app.py --train                  # Train FNO model (v1)
    python app.py --train-v2 --model large # Train FNO v2 model

Examples:
    # Quick test (50 samples, small model)
    python app.py --generate --samples 50 --workers 4
    python app.py --train-v2 --model small --epochs 50

    # Good quality (500 samples, large model)
    python app.py --generate --samples 500 --workers 8
    python app.py --train-v2 --model large --epochs 500

    # Maximum quality (2000 samples, xlarge + specboost)
    python app.py --generate --samples 2000 --workers 8
    python app.py --train-v2 --model xlarge --specboost --epochs 1000

Notes:
    - Batch size auto-detects optimal value based on GPU memory
    - Multi-GPU is automatically detected and used (DataParallel)
    - For xlarge+specboost on 32GB GPUs, batch_size will be auto-reduced
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Proppant Dynamics Hybrid Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                                    # Run web UI
  python app.py --generate --samples 500           # Generate 500 samples
  python app.py --train                            # Train FNO v1
  python app.py --train-v2 --model large           # Train FNO v2 (large)
  python app.py --train-v2 --model xlarge --specboost  # Maximum quality
        """
    )

    # Mode selection
    parser.add_argument('--generate', action='store_true', help='Generate training data')
    parser.add_argument('--train', action='store_true', help='Train FNO model (v1)')
    parser.add_argument('--train-v2', action='store_true', help='Train FNO v2 model (recommended)')

    # Generation options
    parser.add_argument('--samples', '-n', type=int, default=500,
                        help='Number of samples to generate (default: 500)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Parallel workers for generation (default: 4, use -1 for all CPUs)')
    parser.add_argument('--grid', type=str, default='64x32',
                        help='Grid size NxM (default: 64x32)')
    parser.add_argument('--time', '-T', type=float, default=100.0,
                        help='Simulation time [s] (default: 100)')
    parser.add_argument('--dt', type=float, default=4.0,
                        help='Output time step [s] (default: 4)')

    # Training options (v1 and v2)
    parser.add_argument('--epochs', '-e', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--batch-size', '-b', type=int, default=0,
                        help='Batch size (0 = auto-detect based on GPU memory)')

    # FNO v2 specific options
    parser.add_argument('--model', '-m', type=str, default='large',
                        choices=['small', 'medium', 'large', 'xlarge'],
                        help='Model size for v2 (default: large)')
    parser.add_argument('--specboost', action='store_true',
                        help='Use SpecBoost for maximum accuracy (slower)')

    # Web app options
    parser.add_argument('--port', '-p', type=int, default=8050,
                        help='Web app port (default: 8050)')
    parser.add_argument('--debug', action='store_true',
                        help='Run web app in debug mode')

    args = parser.parse_args()

    if args.generate:
        # ===== GENERATE DATA =====
        import os
        from src.training.dataset import generate_dataset

        data_dir = Path(__file__).parent / 'data' / 'processed'

        # Parse grid size
        try:
            nx, ny = map(int, args.grid.split('x'))
        except:
            print(f"Invalid grid format: {args.grid}. Use NxM (e.g., 64x32)")
            return

        # Workers
        n_workers = args.workers
        if n_workers == -1:
            n_workers = os.cpu_count() or 1

        print("=" * 60)
        print("GENERATING TRAINING DATA")
        print("=" * 60)
        print(f"  Samples:  {args.samples}")
        print(f"  Grid:     {nx}x{ny}")
        print(f"  Time:     {args.time}s (dt={args.dt}s)")
        print(f"  Workers:  {n_workers}")
        print(f"  Output:   {data_dir}")
        print("=" * 60)

        generate_dataset(
            output_dir=data_dir,
            n_samples=args.samples,
            n_workers=n_workers,
            grid_size=(nx, ny),
            T=args.time,
            dT=args.dt,
        )

    elif args.train:
        # ===== TRAIN V1 =====
        from src.training.train import main as train_main

        print("=" * 60)
        print("TRAINING FNO v1")
        print("=" * 60)
        print(f"  Epochs:   {args.epochs}")
        print(f"  LR:       {args.lr}")
        print(f"  Patience: {args.patience}")
        print("=" * 60)

        train_main(epochs=args.epochs, lr=args.lr, patience=args.patience)

    elif args.train_v2:
        # ===== TRAIN V2 =====
        from src.training.train_v2 import main as train_v2_main

        print("=" * 60)
        print("TRAINING FNO v2 (Enhanced)")
        print("=" * 60)
        print(f"  Model:    {args.model}" + (" + SpecBoost" if args.specboost else ""))
        print(f"  Epochs:   {args.epochs}")
        batch_str = "auto" if args.batch_size == 0 else str(args.batch_size)
        print(f"  Batch:    {batch_str}")
        print(f"  LR:       {args.lr}")
        print(f"  Patience: {args.patience}")
        print("=" * 60)

        train_v2_main(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            model_size=args.model,
            use_specboost=args.specboost,
            patience=args.patience,
        )

    else:
        # ===== RUN WEB APP =====
        from src.visualization.app import run_app

        print("=" * 60)
        print("PROPPANT TRANSPORT SIMULATOR")
        print("=" * 60)
        print(f"  URL:  http://localhost:{args.port}")
        print("=" * 60)

        run_app(debug=args.debug, port=args.port)


if __name__ == '__main__':
    main()
