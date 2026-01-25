#!/usr/bin/env python3
"""
Main entry point for Proppant Dynamics Hybrid Simulator.

Usage:
    python app.py              # Run web app
    python app.py --generate   # Generate training data
    python app.py --train      # Train FNO model
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Proppant Dynamics Hybrid Simulator")
    parser.add_argument('--generate', action='store_true', help='Generate training data')
    parser.add_argument('--train', action='store_true', help='Train FNO model')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--port', type=int, default=8050, help='Web app port')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    if args.generate:
        from src.training.dataset import generate_dataset
        data_dir = Path(__file__).parent / 'data' / 'processed'
        print(f"Generating {args.samples} samples...")
        generate_dataset(data_dir, n_samples=args.samples)

    elif args.train:
        from src.training.train import main as train_main
        print(f"Training for {args.epochs} epochs...")
        train_main(epochs=args.epochs)

    else:
        from src.visualization.app import run_app
        run_app(debug=args.debug, port=args.port)


if __name__ == '__main__':
    main()
