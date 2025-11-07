import argparse
import csv
import os
from typing import List

import numpy as np

from yoloplay.detectors import KMeansClassifier


def load_csv_data(csv_path: str, normal_label: int = 0) -> List[np.ndarray]:
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


def train_kmeans_model(
    csv_path: str,
    model_path: str = "models/kmeans_classifier.pkl",
    n_clusters: int = 5,
    distance_threshold: float = 0.5,
    random_state: int = 42,
    normal_label: int = 0,
) -> None:
    """
    Train K-Means classifier model from CSV data.

    Args:
        csv_path: Path to CSV training data
        model_path: Path to save trained model
        n_clusters: Number of clusters to create
        distance_threshold: Distance threshold for anomaly detection
        random_state: Random state for reproducibility
        normal_label: Label for normal samples
    """
    print(f"Loading training data from {csv_path}")

    # Load normal keypoints data
    normal_keypoints = load_csv_data(csv_path, normal_label)

    if len(normal_keypoints) == 0:
        raise ValueError(f"No normal samples found with label {normal_label}")

    # Train the model
    classifier = KMeansClassifier(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        random_state=random_state,
    )
    classifier.train(normal_keypoints)

    # Save the model
    classifier.save(model_path)

    # Test on training data
    print("\nTesting model on training data:")
    normal_count = 0
    anomaly_count = 0

    for keypoints in normal_keypoints[:10]:  # Test on first 10 samples
        is_anomaly, score = classifier.detect_anomaly(keypoints)
        if is_anomaly:
            anomaly_count += 1
        else:
            normal_count += 1
        print(f"  Sample: anomaly={is_anomaly}, score={score:.4f}")

    print(f"Results: {normal_count} normal, {anomaly_count} anomalies detected")


def main():
    """Command line interface for training K-Means classifier."""
    parser = argparse.ArgumentParser(description="Train K-Means classifier model")
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to CSV training data file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/kmeans_classifier.pkl",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=5, help="Number of clusters (default: 5)"
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.5,
        help="Distance threshold for anomaly detection (default: 0.5)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--normal-label", type=int, default=0, help="Label value for normal samples"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.n_clusters <= 0:
        parser.error("--n-clusters must be positive")
    if args.distance_threshold <= 0:
        parser.error("--distance-threshold must be positive")

    # Train the model
    train_kmeans_model(
        csv_path=args.csv,
        model_path=args.model_path,
        n_clusters=args.n_clusters,
        distance_threshold=args.distance_threshold,
        random_state=args.random_state,
        normal_label=args.normal_label,
    )


if __name__ == "__main__":
    main()