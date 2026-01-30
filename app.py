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
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience (overrides config)')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint (e.g., "last.pt" or "best.pt")')
    parser.add_argument('--port', type=int, default=8050, help='Web app port')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Web app host (use 0.0.0.0 for network access)')
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
        import json
        config_path = Path(args.config)
        resume_from = None
        if args.resume:
            resume_from = args.resume

        if args.epochs or args.lr or args.patience:
            with open(config_path) as f:
                cfg = json.load(f)
            if args.epochs:
                cfg['training']['n_epochs'] = args.epochs
            if args.lr:
                cfg['training']['optimizer']['lr'] = args.lr
            if args.patience:
                cfg['training']['trainer']['patience'] = args.patience
            import tempfile
            import os
            tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir=config_path.parent)
            json.dump(cfg, tmp_file, indent=4)
            tmp_file.close()
            config_path = Path(tmp_file.name)
            try:
                train_main(config_path=config_path, resume_from=resume_from)
            finally:
                os.unlink(tmp_file.name)
        else:
            train_main(config_path=config_path, resume_from=resume_from)

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
