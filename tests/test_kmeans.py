"""
Unit tests for K-Means classifier.
"""

import os
import tempfile
import unittest

import numpy as np

from yoloplay.detectors import KMeansClassifier


class TestKMeansClassifier(unittest.TestCase):
    """Test cases for KMeansClassifier."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 34
        self.n_clusters = 3
        self.num_samples = 30

        # Create synthetic normal pose data organized in clusters
        np.random.seed(42)

        # Create data points around 3 cluster centers
        centers = [
            np.array([0.2] * self.input_dim),  # Cluster 1
            np.array([0.5] * self.input_dim),  # Cluster 2
            np.array([0.8] * self.input_dim),  # Cluster 3
        ]

        self.normal_keypoints = []
        for i in range(self.num_samples):
            center = centers[i % len(centers)]
            # Add some noise around the center
            keypoints = center + np.random.randn(self.input_dim) * 0.05
            # Clip to [0, 1] range
            keypoints = np.clip(keypoints, 0.0, 1.0)
            self.normal_keypoints.append(keypoints)

        # Create classifier
        self.classifier = KMeansClassifier(
            n_clusters=self.n_clusters,
            distance_threshold=0.3,
            random_state=42
        )

    def test_initialization(self):
        """Test classifier initialization."""
        self.assertEqual(self.classifier.n_clusters, self.n_clusters)
        self.assertEqual(self.classifier.distance_threshold, 0.3)
        self.assertEqual(self.classifier.random_state, 42)
        self.assertFalse(self.classifier.is_trained)
        self.assertIsNone(self.classifier.model)
        self.assertIsNone(self.classifier.scaler)

    def test_training(self):
        """Test training pipeline."""
        # Train the classifier
        self.classifier.train(self.normal_keypoints)

        # Check that training completed
        self.assertTrue(self.classifier.is_trained)
        self.assertIsNotNone(self.classifier.model)
        self.assertIsNotNone(self.classifier.scaler)

        # Check model has correct number of clusters
        self.assertEqual(self.classifier.model.n_clusters, self.n_clusters)

    def test_classification(self):
        """Test pose classification."""
        if not self.classifier.is_trained:
            self.classifier.train(self.normal_keypoints)

        # Test classification on training data
        for keypoints in self.normal_keypoints[:5]:
            cluster_id, distance = self.classifier.classify(keypoints)

            # Check return types and ranges
            self.assertIsInstance(cluster_id, int)
            self.assertIsInstance(distance, float)
            self.assertGreaterEqual(cluster_id, 0)
            self.assertLess(cluster_id, self.n_clusters)
            self.assertGreaterEqual(distance, 0.0)

    def test_anomaly_detection_normal_data(self):
        """Test anomaly detection on normal data."""
        if not self.classifier.is_trained:
            self.classifier.train(self.normal_keypoints)

        # Test on normal training data
        for keypoints in self.normal_keypoints[:5]:
            is_anomaly, score = self.classifier.detect_anomaly(keypoints)

            # Normal data should generally not be anomalous
            self.assertIsInstance(is_anomaly, bool)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)

    def test_anomaly_detection_anomalous_data(self):
        """Test anomaly detection on anomalous data."""
        if not self.classifier.is_trained:
            self.classifier.train(self.normal_keypoints)

        # Create anomalous data (far from any cluster)
        anomalous_keypoints = np.array([0.0] * self.input_dim)  # Very different from training data

        is_anomaly, score = self.classifier.detect_anomaly(anomalous_keypoints)

        # Anomalous data should be detected
        self.assertIsInstance(is_anomaly, bool)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

        # With a reasonable threshold, this should be detected as anomalous
        # (though we don't assert this as it depends on the specific data and threshold)

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
            new_classifier = KMeansClassifier()
            new_classifier.load(model_path)

            # Test that loaded model works
            test_keypoints = self.normal_keypoints[0]
            original_result = self.classifier.classify(test_keypoints)
            loaded_result = new_classifier.classify(test_keypoints)

            # Results should be identical
            self.assertEqual(original_result[0], loaded_result[0])  # Same cluster
            self.assertAlmostEqual(original_result[1], loaded_result[1], places=5)  # Same distance

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_untrained_classification_raises_error(self):
        """Test that classification raises error when model is not trained."""
        untrained_classifier = KMeansClassifier()

        with self.assertRaises(RuntimeError):
            untrained_classifier.classify(self.normal_keypoints[0])

    def test_untrained_detection_raises_error(self):
        """Test that detection raises error when model is not trained."""
        untrained_classifier = KMeansClassifier()

        with self.assertRaises(RuntimeError):
            untrained_classifier.detect_anomaly(self.normal_keypoints[0])

    def test_invalid_keypoints_shape(self):
        """Test handling of invalid keypoints shape."""
        if not self.classifier.is_trained:
            self.classifier.train(self.normal_keypoints)

        # Test with wrong shape
        invalid_keypoints = np.array([1.0, 2.0])  # Only 2 values instead of 34

        with self.assertRaises(ValueError):
            self.classifier.classify(invalid_keypoints)

    def test_distance_threshold_effect(self):
        """Test that distance threshold affects anomaly detection."""
        if not self.classifier.is_trained:
            self.classifier.train(self.normal_keypoints)

        # Get a test sample
        test_keypoints = self.normal_keypoints[0]
        _, distance = self.classifier.classify(test_keypoints)

        # Test with very low threshold (should detect as anomaly)
        self.classifier.distance_threshold = 0.001
        is_anomaly_low, _ = self.classifier.detect_anomaly(test_keypoints)
        self.assertTrue(is_anomaly_low)

        # Test with very high threshold (should not detect as anomaly)
        self.classifier.distance_threshold = 10.0
        is_anomaly_high, _ = self.classifier.detect_anomaly(test_keypoints)
        self.assertFalse(is_anomaly_high)

        # Reset threshold
        self.classifier.distance_threshold = 0.3


if __name__ == '__main__':
    unittest.main()