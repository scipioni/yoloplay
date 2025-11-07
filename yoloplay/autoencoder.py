"""
Autoencoder-based one-class classifier for pose anomaly detection.

This module implements an autoencoder neural network that learns to reconstruct
normal pose keypoints, combined with one-class SVM on the latent space for
robust anomaly detection.
"""

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class PoseAutoencoder(nn.Module):
    """
    Autoencoder for pose keypoints reconstruction.

    Architecture:
    - Encoder: Compresses 34-dimensional pose keypoints to latent space
    - Decoder: Reconstructs pose keypoints from latent representation
    """

    def __init__(self, input_dim: int = 34, latent_dim: int = 16, hidden_dims: List[int] = None):
        """
        Initialize the autoencoder.

        Args:
            input_dim: Input dimension (34 for 17 keypoints * 2 coordinates)
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions for encoder/decoder
        """
        super(PoseAutoencoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Encoder layers
        encoder_layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim

        # Final encoder layer to latent space
        encoder_layers.append(nn.Linear(current_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers (reverse of encoder)
        decoder_layers = []
        current_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim

        # Final decoder layer to output
        decoder_layers.append(nn.Linear(current_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (reconstructed_output, latent_representation)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation of input."""
        with torch.no_grad():
            return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        with torch.no_grad():
            return self.decoder(latent)


class OneClassAutoencoderClassifier:
    """
    One-Class Classifier using Autoencoder + One-Class SVM.

    The autoencoder learns to reconstruct normal poses, and one-class SVM
    establishes a decision boundary in the latent space for anomaly detection.
    """

    def __init__(
        self,
        input_dim: int = 34,
        latent_dim: int = 16,
        hidden_dims: List[int] = None,
        svm_nu: float = 0.1,
        svm_kernel: str = "rbf",
        svm_gamma: str = "scale",
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
        device: str = "auto"
    ):
        """
        Initialize the one-class autoencoder classifier.

        Args:
            input_dim: Input dimension for pose keypoints
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions
            svm_nu: One-class SVM nu parameter
            svm_kernel: One-class SVM kernel
            svm_gamma: One-class SVM gamma parameter
            learning_rate: Autoencoder learning rate
            batch_size: Training batch size
            num_epochs: Number of training epochs
            device: Device for training ('cpu', 'cuda', or 'auto')
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.svm_nu = svm_nu
        self.svm_kernel = svm_kernel
        self.svm_gamma = svm_gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize components
        self.autoencoder = PoseAutoencoder(input_dim, latent_dim, hidden_dims).to(self.device)
        self.svm = OneClassSVM(nu=svm_nu, kernel=svm_kernel, gamma=svm_gamma)
        self.latent_scaler = StandardScaler()

        self.is_trained = False

    def train_autoencoder(self, keypoints_data: List[np.ndarray]) -> None:
        """
        Train the autoencoder on normal pose keypoints.

        Args:
            keypoints_data: List of keypoints arrays, each shape (34,)
        """
        if not keypoints_data:
            raise ValueError("No training data provided")

        # Convert to numpy array and then tensor
        X = np.array(keypoints_data, dtype=np.float32)
        print(f"Training autoencoder on {X.shape[0]} samples with {X.shape[1]} features")

        # Create dataset and dataloader
        dataset = TensorDataset(torch.from_numpy(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)

        # Training loop
        self.autoencoder.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            for batch in dataloader:
                inputs = batch[0].to(self.device)

                # Forward pass
                outputs, _ = self.autoencoder(inputs)
                loss = criterion(outputs, inputs)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.6f}")

        print("Autoencoder training completed")

    def extract_latent_features(self, keypoints_data: List[np.ndarray]) -> np.ndarray:
        """
        Extract latent features from trained autoencoder.

        Args:
            keypoints_data: List of keypoints arrays

        Returns:
            Latent features array
        """
        if not keypoints_data:
            return np.array([])

        X = np.array(keypoints_data, dtype=np.float32)
        X_tensor = torch.from_numpy(X).to(self.device)

        self.autoencoder.eval()
        with torch.no_grad():
            _, latent = self.autoencoder(X_tensor)

        return latent.cpu().numpy()

    def train_svm(self, latent_features: np.ndarray) -> None:
        """
        Train one-class SVM on latent features.

        Args:
            latent_features: Latent features from autoencoder
        """
        if latent_features.size == 0:
            raise ValueError("No latent features provided for SVM training")

        print(f"Training SVM on {latent_features.shape[0]} latent samples with {latent_features.shape[1]} features")

        # Standardize latent features
        latent_scaled = self.latent_scaler.fit_transform(latent_features)

        # Train SVM
        self.svm.fit(latent_scaled)
        print("SVM training completed")

    def train(self, keypoints_data: List[np.ndarray]) -> None:
        """
        Train the complete pipeline: autoencoder + one-class SVM.

        Args:
            keypoints_data: List of keypoints arrays for normal poses
        """
        # Train autoencoder
        self.train_autoencoder(keypoints_data)

        # Extract latent features
        latent_features = self.extract_latent_features(keypoints_data)

        # Train SVM on latent space
        self.train_svm(latent_features)

        self.is_trained = True
        print("One-class autoencoder classifier training completed")

    def detect(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if keypoints are anomalous.

        Args:
            keypoints: Keypoints array, shape (34,)

        Returns:
            Tuple of (is_anomaly, anomaly_score)
            - is_anomaly: True if anomalous, False if normal
            - anomaly_score: Combined score from reconstruction error and SVM
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before detection")

        # Ensure correct shape
        keypoints = np.array(keypoints, dtype=np.float32).flatten()
        if keypoints.shape != (self.input_dim,):
            raise ValueError(f"Expected {self.input_dim} keypoints, got {keypoints.shape}")

        # Get reconstruction and latent representation
        keypoints_tensor = torch.from_numpy(keypoints).unsqueeze(0).to(self.device)
        self.autoencoder.eval()

        with torch.no_grad():
            reconstructed, latent = self.autoencoder(keypoints_tensor)

        # Calculate reconstruction error
        reconstruction_error = torch.mean((reconstructed - keypoints_tensor) ** 2).item()

        # SVM prediction on latent space
        latent_np = latent.cpu().numpy()
        latent_scaled = self.latent_scaler.transform(latent_np)
        svm_prediction = self.svm.predict(latent_scaled)[0]  # 1 for normal, -1 for anomaly
        svm_score = self.svm.decision_function(latent_scaled)[0]

        # Combine scores: reconstruction error + SVM decision function
        # Normalize reconstruction error to similar scale as SVM score
        combined_score = reconstruction_error * 10.0 + svm_score

        # Anomaly if SVM predicts anomaly OR reconstruction error is high
        is_anomaly = (svm_prediction == -1) or (reconstruction_error > 0.1)

        return is_anomaly, float(combined_score)

    def save(self, model_path: str) -> None:
        """
        Save the trained model to file.

        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        os.makedirs(
            os.path.dirname(model_path) if os.path.dirname(model_path) else ".",
            exist_ok=True,
        )

        # Convert autoencoder state dict to CPU for saving
        autoencoder_state = {k: v.cpu() for k, v in self.autoencoder.state_dict().items()}

        model_data = {
            "autoencoder_state": autoencoder_state,
            "autoencoder_config": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "hidden_dims": self.hidden_dims,
            },
            "svm": self.svm,
            "latent_scaler": self.latent_scaler,
            "svm_params": {
                "nu": self.svm_nu,
                "kernel": self.svm_kernel,
                "gamma": self.svm_gamma,
            },
            "training_params": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
            },
            "is_trained": self.is_trained,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"One-class autoencoder classifier saved to {model_path}")

    def load(self, model_path: str) -> None:
        """
        Load a trained model from file.

        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Restore autoencoder
        config = model_data["autoencoder_config"]
        self.autoencoder = PoseAutoencoder(
            config["input_dim"],
            config["latent_dim"],
            config["hidden_dims"]
        ).to(self.device)

        # Load state dict
        state_dict = model_data["autoencoder_state"]
        self.autoencoder.load_state_dict(state_dict)

        # Restore SVM and scaler
        self.svm = model_data["svm"]
        self.latent_scaler = model_data["latent_scaler"]

        # Restore parameters
        svm_params = model_data["svm_params"]
        self.svm_nu = svm_params["nu"]
        self.svm_kernel = svm_params["kernel"]
        self.svm_gamma = svm_params["gamma"]

        training_params = model_data["training_params"]
        self.learning_rate = training_params["learning_rate"]
        self.batch_size = training_params["batch_size"]
        self.num_epochs = training_params["num_epochs"]

        self.is_trained = model_data["is_trained"]

        print(f"One-class autoencoder classifier loaded from {model_path}")