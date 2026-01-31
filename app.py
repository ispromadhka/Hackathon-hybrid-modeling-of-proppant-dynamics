#!/usr/bin/env python3
"""
Main entry point for Proppant Dynamics Hybrid Simulator.

Usage:
    python app.py              # Run new FastAPI web app
    python app.py --legacy     # Run legacy Dash interface
    python app.py --generate   # Generate training data
    python app.py --train      # Train FNO model
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Proppant Dynamics Hybrid Simulator")
    parser.add_argument('--generate', action='store_true', help='Generate training data')
    parser.add_argument('--train', action='store_true', help='Train FNO model')
    parser.add_argument('--legacy', action='store_true', help='Use legacy Dash interface')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples to generate')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers for generation (-1 = all CPUs)')
    parser.add_argument('--config', type=str, default='configs/default.json', help='Config file path')
    parser.add_argument('--clear-processed', action='store_true', help='Clear data/processed before generating')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for fine-tuning')
    parser.add_argument('--port', type=int, default=8050, help='Web app port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web app host')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    if args.generate:
        import os
        from src.training.dataset import generate_dataset
        root = Path(__file__).parent
        data_dir = root / 'data' / 'processed'
        n_workers = args.workers
        if n_workers == -1:
            n_workers = os.cpu_count() or 1
        n = generate_dataset(data_dir, n_samples=args.samples, n_workers=n_workers, config_path=Path(args.config), clear_processed=bool(args.clear_processed))
        print(f"Built {n} samples in {data_dir}")

    elif args.train:
        from src.training.train import main as train_main
        mode = "Resume" if args.resume else "New"
        print(f"[{mode}] Training: {args.epochs} epochs, lr={args.lr}, patience={args.patience}")
        train_main(epochs=args.epochs, lr=args.lr, patience=args.patience, resume=args.resume)

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
