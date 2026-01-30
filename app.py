#!/usr/bin/env python3
"""
Main entry point for Proppant Dynamics Hybrid Simulator.

Usage:
    python app.py                          # Run web app (FastAPI)
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
    parser.add_argument('--legacy', action='store_true', help='Use legacy Dash interface')

    # Generation options
    parser.add_argument('--samples', '-n', type=int, default=500, help='Number of samples to generate')
    parser.add_argument('--workers', '-w', type=int, default=1, help='Parallel workers for generation (-1 = all CPUs)')
    parser.add_argument('--config', type=str, default='configs/default.json', help='Config file path')
    parser.add_argument('--clear-processed', action='store_true', help='Clear data/processed before generating')
    parser.add_argument('--grid', type=str, default='64x32', help='Grid size NxM (default: 64x32)')
    parser.add_argument('--time', '-T', type=float, default=100.0, help='Simulation time [s] (default: 100)')
    parser.add_argument('--dt', type=float, default=4.0, help='Output time step [s] (default: 4)')

    # Training options (v1 and v2)
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--batch-size', '-b', type=int, default=0, help='Batch size (0 = auto-detect)')
    parser.add_argument('--physics-loss', action='store_true', default=True, help='Use physics-informed loss')
    parser.add_argument('--no-physics-loss', action='store_true', help='Disable physics-informed loss')

    # FNO v2 specific options
    parser.add_argument('--model', '-m', type=str, default='large',
                        choices=['small', 'medium', 'large', 'xlarge'],
                        help='Model size for v2 (default: large)')
    parser.add_argument('--specboost', action='store_true', help='Use SpecBoost for maximum accuracy')

    # GPU parallelization options
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training (AMP)')
    parser.add_argument('--no-multi-gpu', action='store_true', help='Disable multi-GPU DataParallel')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loading workers (0 = main process)')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps')

    # Web app options
    parser.add_argument('--port', '-p', type=int, default=8050, help='Web app port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web app host')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    if args.generate:
        # ===== GENERATE DATA =====
        import os
        from src.training.dataset import generate_dataset

        root = Path(__file__).parent
        data_dir = root / 'data' / 'processed'

        # Parse grid size
        try:
            nx, ny = map(int, args.grid.split('x'))
        except ValueError:
            print(f"Invalid grid format: {args.grid}. Use NxM (e.g., 64x32)")
            return

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

        n = generate_dataset(
            data_dir,
            n_samples=args.samples,
            n_workers=n_workers,
            config_path=Path(args.config),
            clear_processed=bool(args.clear_processed),
            grid_size=(nx, ny),
            T=args.time,
            dT=args.dt,
        )
        print(f"Built {n} samples in {data_dir}")

    elif args.train:
        # ===== TRAIN V1 =====
        from src.training.train import main as train_main

        use_physics = args.physics_loss and not args.no_physics_loss
        use_amp = not args.no_amp
        use_multi_gpu = not args.no_multi_gpu

        print("=" * 60)
        print("TRAINING FNO v1")
        print("=" * 60)
        print(f"  Epochs:   {args.epochs}")
        print(f"  LR:       {args.lr}")
        print(f"  Patience: {args.patience}")
        print(f"  Physics loss: {use_physics}")
        print(f"  AMP: {use_amp}, Multi-GPU: {use_multi_gpu}")
        print("=" * 60)

        train_main(
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            use_physics_loss=use_physics,
            use_amp=use_amp,
            use_multi_gpu=use_multi_gpu,
            num_workers=args.num_workers,
            gradient_accumulation=args.grad_accum,
        )

    elif args.train_v2:
        # ===== TRAIN V2 =====
        from src.training.train_v2 import main as train_v2_main

        print("=" * 60)
        print("TRAINING FNO v2 (Enhanced)")
        print("=" * 60)
        print(f"  Model:    {args.model}" + (" + SpecBoost" if args.specboost else ""))
        print(f"  Epochs:   {args.epochs}")
        print(f"  LR:       {args.lr}")
        print(f"  Patience: {args.patience}")
        print(f"  Batch:    {'auto' if args.batch_size == 0 else args.batch_size}")
        print("=" * 60)

        train_v2_main(
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            model_size=args.model,
            use_specboost=args.specboost,
            batch_size=args.batch_size if args.batch_size > 0 else None,
        )

    elif args.legacy:
        # Legacy Dash interface
        from src.visualization.app import run_app
        run_app(debug=args.debug, port=args.port)

    else:
        # New FastAPI interface
        import uvicorn
        print(f"Starting web server at http://{args.host}:{args.port}")
        uvicorn.run(
            "web.server:app",
            host=args.host,
            port=args.port,
            reload=args.debug
        )


if __name__ == '__main__':
    main()
