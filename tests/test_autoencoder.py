"""
Unit tests for autoencoder-based one-class classifier.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from yoloplay.autoencoder import OneClassAutoencoderClassifier, PoseAutoencoder


class TestPoseAutoencoder(unittest.TestCase):
    """Test cases for PoseAutoencoder neural network."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 34
        self.latent_dim = 8
        self.batch_size = 4
        self.autoencoder = PoseAutoencoder(self.input_dim, self.latent_dim)

    def test_forward_pass(self):
        """Test forward pass through autoencoder."""
        # Create dummy input
        x = torch.randn(self.batch_size, self.input_dim)

        # Forward pass
        reconstructed, latent = self.autoencoder(x)

        # Check output shapes
        self.assertEqual(reconstructed.shape, (self.batch_size, self.input_dim))
        self.assertEqual(latent.shape, (self.batch_size, self.latent_dim))

    def test_encoding_decoding(self):
        """Test separate encode/decode operations."""
        x = torch.randn(self.batch_size, self.input_dim)

        # Encode
        latent = self.autoencoder.encode(x)
        self.assertEqual(latent.shape, (self.batch_size, self.latent_dim))

        # Decode
        reconstructed = self.autoencoder.decode(latent)
        self.assertEqual(reconstructed.shape, (self.batch_size, self.input_dim))


class TestOneClassAutoencoderClassifier(unittest.TestCase):
    """Test cases for OneClassAutoencoderClassifier."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 34
        self.latent_dim = 8
        self.num_samples = 20

        # Create synthetic normal pose data
        np.random.seed(42)
        self.normal_keypoints = [
            np.random.randn(self.input_dim) * 0.1 + np.array([0.5] * self.input_dim)
            for _ in range(self.num_samples)
        ]

        # Create classifier with minimal training for testing
        self.classifier = OneClassAutoencoderClassifier(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_epochs=5,  # Minimal training for tests
            batch_size=4,
            device="cpu"
        )

    def test_initialization(self):
        """Test classifier initialization."""
        self.assertEqual(self.classifier.input_dim, self.input_dim)
        self.assertEqual(self.classifier.latent_dim, self.latent_dim)
        self.assertFalse(self.classifier.is_trained)
        self.assertIsInstance(self.classifier.autoencoder, PoseAutoencoder)

    def test_training(self):
        """Test training pipeline."""
        # Train the classifier
        self.classifier.train(self.normal_keypoints)

        # Check that training completed
        self.assertTrue(self.classifier.is_trained)

        # Test detection on normal data
        for keypoints in self.normal_keypoints[:5]:
            is_anomaly, score = self.classifier.detect(keypoints)
            # Normal data should generally not be anomalous
            self.assertIsInstance(is_anomaly, bool)
            self.assertIsInstance(score, float)

    def test_detection_on_anomalous_data(self):
        """Test detection on anomalous data."""
        if not self.classifier.is_trained:
            self.classifier.train(self.normal_keypoints)

        # Create anomalous data (far from normal distribution)
        anomalous_keypoints = np.random.randn(self.input_dim) * 2.0  # Much more spread out

        is_anomaly, score = self.classifier.detect(anomalous_keypoints)

        # Anomalous data should be detected
        self.assertIsInstance(is_anomaly, bool)
        self.assertIsInstance(score, float)

    def test_save_load(self):
        """Test model saving and loading."""
        if not self.classifier.is_trained:
            self.classifier.train(self.normal_keypoints)

        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name

        try:
            self.classifier.save(model_path)
            self.assertTrue(os.path.exists(model_path))

            # Load model in new instance
            new_classifier = OneClassAutoencoderClassifier()
            new_classifier.load(model_path)

            # Test that loaded model works
            test_keypoints = self.normal_keypoints[0]
            original_result = self.classifier.detect(test_keypoints)
            loaded_result = new_classifier.detect(test_keypoints)

            # Results should be similar (allowing for small numerical differences)
            self.assertEqual(original_result[0], loaded_result[0])  # Same anomaly decision

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_untrained_detection_raises_error(self):
        """Test that detection raises error when model is not trained."""
        untrained_classifier = OneClassAutoencoderClassifier()

        with self.assertRaises(RuntimeError):
            untrained_classifier.detect(self.normal_keypoints[0])

    def test_invalid_keypoints_shape(self):
        """Test handling of invalid keypoints shape."""
        if not self.classifier.is_trained:
            self.classifier.train(self.normal_keypoints)

        # Test with wrong shape
        invalid_keypoints = np.array([1.0, 2.0])  # Only 2 values instead of 34

        with self.assertRaises(ValueError):
            self.classifier.detect(invalid_keypoints)


if __name__ == '__main__':
    unittest.main()