import argparse
import csv
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class OneClassSVMAnomalyDetector:
    """
    One-Class SVM anomaly detector for pose keypoints.
    Trained on normal poses (label=0) to detect anomalies.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "scale",
    ):
        """
        Initialize the SVM anomaly detector.

        Args:
            model_path: Path to saved model file
            nu: Anomaly parameter (0 < nu <= 1), upper bound on fraction of outliers
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            gamma: Kernel coefficient ('scale', 'auto', or float)
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
        self.scaler = None
        self.is_trained = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, keypoints_data: List[np.ndarray]) -> None:
        """
        Train the One-Class SVM on normal keypoints data.

        Args:
            keypoints_data: List of keypoints arrays, each shape (34,)
        """
        if not keypoints_data:
            raise ValueError("No training data provided")

        # Convert to numpy array
        X = np.array(keypoints_data)
        print(f"Training SVM on {X.shape[0]} samples with {X.shape[1]} features")

        # Standardize the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Initialize and train the model
        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.model.fit(X_scaled)
        self.is_trained = True

        print(
            f"SVM training completed. Nu={self.nu}, kernel={self.kernel}, gamma={self.gamma}"
        )

    def detect(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if keypoints are anomalous.

        Args:
            keypoints: Keypoints array, shape (34,)

        Returns:
            Tuple of (is_anomaly, anomaly_score)
            - is_anomaly: True if anomalous, False if normal
            - anomaly_score: Negative score for normal, positive for anomalies
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before detection")

        # Ensure correct shape
        if keypoints.shape != (34,):
            keypoints = keypoints.flatten()
            if keypoints.shape != (34,):
                raise ValueError(f"Expected 34 keypoints, got {keypoints.shape}")

        # Standardize and predict
        keypoints_scaled = self.scaler.transform([keypoints])
        prediction = self.model.predict(keypoints_scaled)[
            0
        ]  # 1 for normal, -1 for anomaly
        score = self.model.decision_function(keypoints_scaled)[0]

        is_anomaly = prediction == -1
        return is_anomaly, float(score)

    def save(self, model_path: str) -> None:
        """
        Save the trained model and scaler to file.

        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        os.makedirs(
            os.path.dirname(model_path) if os.path.dirname(model_path) else ".",
            exist_ok=True,
        )

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "is_trained": self.is_trained,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"SVM model saved to {model_path}")

    def load(self, model_path: str) -> None:
        """
        Load a trained model and scaler from file.

        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.nu = model_data["nu"]
        self.kernel = model_data["kernel"]
        self.gamma = model_data["gamma"]
        self.is_trained = model_data["is_trained"]

        print(f"SVM model loaded from {model_path}")


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


def train_svm_model(
    csv_path: str,
    model_path: str = "models/svm_anomaly_detector.pkl",
    nu: float = 0.1,
    kernel: str = "rbf",
    gamma: str = "scale",
    normal_label: int = 0,
    grid_search: bool = False,
) -> None:
    """
    Train SVM anomaly detection model from CSV data.

    Args:
        csv_path: Path to CSV training data
        model_path: Path to save trained model
        nu: Anomaly parameter
        kernel: Kernel type
        gamma: Kernel coefficient
        normal_label: Label for normal samples
        grid_search: Whether to perform grid search for hyperparameters
    """
    print(f"Loading training data from {csv_path}")

    # Load normal keypoints data
    normal_keypoints = load_csv_data(csv_path, normal_label)

    if len(normal_keypoints) == 0:
        raise ValueError(f"No normal samples found with label {normal_label}")

    # Perform grid search if requested
    if grid_search:
        print("Performing grid search for hyperparameters...")
        X = np.array(normal_keypoints)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        param_grid = {
            "nu": [0.05, 0.1, 0.15, 0.2],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto", 0.1, 0.01],
        }

        svm = OneClassSVM()
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_scaled)

        print(f"Best parameters: {grid_search.best_params_}")
        nu = grid_search.best_params_["nu"]
        kernel = grid_search.best_params_["kernel"]
        gamma = grid_search.best_params_["gamma"]

    # Train the model
    detector = OneClassSVMAnomalyDetector(nu=nu, kernel=kernel, gamma=gamma)
    detector.train(normal_keypoints)

    # Save the model
    detector.save(model_path)

    # Test on training data
    print("\nTesting model on training data:")
    normal_count = 0
    anomaly_count = 0

    for keypoints in normal_keypoints[:10]:  # Test on first 10 samples
        is_anomaly, score = detector.detect(keypoints)
        if is_anomaly:
            anomaly_count += 1
        else:
            normal_count += 1
        print(f"  Sample: anomaly={is_anomaly}, score={score:.4f}")

    print(f"Results: {normal_count} normal, {anomaly_count} anomalies detected")


def main():
    """Command line interface for training SVM anomaly detector."""
    parser = argparse.ArgumentParser(description="Train SVM anomaly detection model")
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to CSV training data file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/svm_anomaly_detector.pkl",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--nu", type=float, default=0.1, help="Anomaly parameter (0 < nu <= 1)"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly", "sigmoid"],
        help="Kernel type",
    )
    parser.add_argument("--gamma", type=str, default="scale", help="Kernel coefficient")
    parser.add_argument(
        "--normal-label", type=int, default=0, help="Label value for normal samples"
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Perform grid search for hyperparameters",
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0 < args.nu <= 1:
        parser.error("--nu must be between 0 and 1")

    # Train the model
    train_svm_model(
        csv_path=args.csv,
        model_path=args.model_path,
        nu=args.nu,
        kernel=args.kernel,
        gamma=args.gamma,
        normal_label=args.normal_label,
        grid_search=args.grid_search,
    )


if __name__ == "__main__":
    main()
