#!/usr/bin/env python3
"""
Training script for one-class autoencoder classifier.

This script trains an autoencoder on normal pose keypoints and integrates
one-class SVM for anomaly detection.
"""

import argparse
import csv
import os
from typing import List

import numpy as np

from yoloplay.autoencoder import OneClassAutoencoderClassifier
from yoloplay.svm import load_csv_data


def load_csv_keypoints(csv_path: str, normal_label: int = 0) -> List[np.ndarray]:
    """
    Load keypoints data from CSV file, filtering for normal samples.

    Args:
        csv_path: Path to CSV file
        normal_label: Label value for normal samples (default: 0)

    Returns:
        List of keypoints arrays for normal samples
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    normal_keypoints = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row_num, row in enumerate(reader, 1):
            if len(row) != 35:  # 1 label + 17*2 coordinates
                print(f"Warning: Skipping row {row_num} - invalid length {len(row)}")
                continue

            try:
                label = int(row[0])
                if label == normal_label:
                    keypoints = np.array([float(x) for x in row[1:]])
                    normal_keypoints.append(keypoints)
            except ValueError as e:
                print(f"Warning: Skipping row {row_num} - parsing error: {e}")

    print(f"Loaded {len(normal_keypoints)} normal samples from {csv_path}")
    return normal_keypoints


def train_autoencoder_model(
    csv_path: str,
    model_path: str = "models/autoencoder_anomaly_detector.pkl",
    latent_dim: int = 16,
    hidden_dims: List[int] = None,
    svm_nu: float = 0.1,
    svm_kernel: str = "rbf",
    svm_gamma: str = "scale",
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    num_epochs: int = 100,
    normal_label: int = 0,
    device: str = "auto"
) -> None:
    """
    Train autoencoder anomaly detection model from CSV data.

    Args:
        csv_path: Path to CSV training data
        model_path: Path to save trained model
        latent_dim: Dimension of latent space
        hidden_dims: Hidden layer dimensions
        svm_nu: One-class SVM nu parameter
        svm_kernel: One-class SVM kernel
        svm_gamma: One-class SVM gamma parameter
        learning_rate: Autoencoder learning rate
        batch_size: Training batch size
        num_epochs: Number of training epochs
        normal_label: Label for normal samples
        device: Device for training
    """
    print(f"Loading training data from {csv_path}")

    # Load normal keypoints data
    normal_keypoints = load_csv_keypoints(csv_path, normal_label)

    if len(normal_keypoints) == 0:
        raise ValueError(f"No normal samples found with label {normal_label}")

    # Initialize the classifier
    classifier = OneClassAutoencoderClassifier(
        input_dim=34,  # 17 keypoints * 2 coordinates
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        svm_nu=svm_nu,
        svm_kernel=svm_kernel,
        svm_gamma=svm_gamma,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device
    )

    # Train the complete pipeline
    classifier.train(normal_keypoints)

    # Save the model
    classifier.save(model_path)

    # Test on training data
    print("\nTesting model on training data:")
    normal_count = 0
    anomaly_count = 0

    for keypoints in normal_keypoints[:10]:  # Test on first 10 samples
        is_anomaly, score = classifier.detect(keypoints)
        if is_anomaly:
            anomaly_count += 1
        else:
            normal_count += 1
        print(f"  Sample: anomaly={is_anomaly}, score={score:.4f}")

    print(f"Results: {normal_count} normal, {anomaly_count} anomalies detected")


def main():
    """Command line interface for training autoencoder anomaly detector."""
    parser = argparse.ArgumentParser(description="Train one-class autoencoder anomaly detection model")
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to CSV training data file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/autoencoder_anomaly_detector.pkl",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=16, help="Dimension of latent space"
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer dimensions for encoder/decoder",
    )
    parser.add_argument(
        "--svm-nu", type=float, default=0.1, help="One-class SVM nu parameter (0 < nu <= 1)"
    )
    parser.add_argument(
        "--svm-kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly", "sigmoid"],
        help="One-class SVM kernel type",
    )
    parser.add_argument(
        "--svm-gamma", type=str, default="scale", help="One-class SVM gamma parameter"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Autoencoder learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--normal-label", type=int, default=0, help="Label value for normal samples"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training",
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0 < args.svm_nu <= 1:
        parser.error("--svm-nu must be between 0 and 1")

    if args.latent_dim <= 0:
        parser.error("--latent-dim must be positive")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")

    # Train the model
    train_autoencoder_model(
        csv_path=args.csv,
        model_path=args.model_path,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        svm_nu=args.svm_nu,
        svm_kernel=args.svm_kernel,
        svm_gamma=args.svm_gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        normal_label=args.normal_label,
        device=args.device,
    )


if __name__ == "__main__":
    main()