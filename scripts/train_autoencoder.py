#!/usr/bin/env python3
"""
Train an autoencoder for one-class anomaly detection on standing poses.
Usage: python scripts/train_autoencoder.py --csv data/keypoints.csv --output models/autoencoder.pt
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yoloplay.classification import train_autoencoder


def main():
    parser = argparse.ArgumentParser(
        description="Train autoencoder for one-class anomaly detection on standing poses"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file with keypoints (only standing poses with label=0 will be used)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/pose_autoencoder.pt",
        help="Path to save the trained model (default: models/pose_autoencoder.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on: 'auto', 'cpu', or 'cuda' (default: auto)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/validation split ratio (default: 0.8 = 80%% train, 20%% val)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (default: 15)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for regularization (default: 0.1)",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=[64, 32, 16],
        help="Hidden layer sizes for encoder (default: 64 32 16)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0 < args.split_ratio < 1:
        parser.error("--split-ratio must be between 0 and 1")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be between 0 and 1")
    if args.patience < 1:
        parser.error("--patience must be at least 1")
    if not os.path.exists(args.csv):
        parser.error(f"CSV file not found: {args.csv}")

    # Train the autoencoder
    print(f"Training autoencoder with the following configuration:")
    print(f"  CSV file: {args.csv}")
    print(f"  Output path: {args.output}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Hidden sizes: {args.hidden_sizes}")
    print(f"  Device: {args.device}")
    print(f"  Split ratio: {args.split_ratio}")
    print(f"  Early stopping patience: {args.patience}")
    print()

    train_autoencoder(
        csv_file=args.csv,
        model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        split_ratio=args.split_ratio,
        patience=args.patience,
        random_seed=args.random_seed,
        dropout_rate=args.dropout,
        hidden_sizes=args.hidden_sizes,
    )


if __name__ == "__main__":
    main()