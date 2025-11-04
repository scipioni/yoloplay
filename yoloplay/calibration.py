"""
Calibration module for collecting keypoints during pose detection.
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import time
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class Calibration:
    """
    Class to collect and store keypoints during calibration mode.
    """

    def __init__(self):
        self.keypoints: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def add_keypoints(self, keypoints_data) -> None:
        """
        Add keypoints data to the calibration collection.

        Args:
            keypoints_data: The keypoints data from pose detection
        """
        if keypoints_data is not None and keypoints_data.data is not None:
            try:
                # Get keypoints data (already normalized)
                person_kpts = keypoints_data.data

                # Convert to list ensuring it's JSON serializable
                person_kpts_list = (
                    person_kpts.tolist()
                    if hasattr(person_kpts, "tolist")
                    else person_kpts
                )

                entry = {
                    "timestamp": time.time() - self.start_time,
                    "keypoints": {
                        "count": int(person_kpts.shape[0]),
                        "data": person_kpts_list,
                        "format": f"{keypoints_data.source.upper()} keypoints (normalized x, y, confidence)",
                    }
                }

                self.keypoints.append(entry)
            except Exception as e:
                print(f"Error adding keypoints: {e}")

    # def save_to_file(self, filename: str = "calibration_data.json") -> None:
    #     """
    #     Save collected keypoints to a JSON file.

    #     Args:
    #         filename: Name of the file to save to
    #     """
    #     with open(filename, 'w') as f:
    #         json.dump(self.keypoints, f, indent=2)
    #     print(f"Calibration data saved to {filename}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the collected calibration data.

        Returns:
            Dictionary with summary statistics
        """
        if not self.keypoints:
            return {"total_frames": 0, "duration": 0.0}

        total_frames = len(self.keypoints)
        duration = self.keypoints[-1]["timestamp"] if self.keypoints else 0.0

        return {
            "total_frames": total_frames,
            "duration": duration,
            "avg_keypoints_per_frame": sum(kp["keypoints"]["count"] for kp in self.keypoints) / total_frames if total_frames > 0 else 0
        }

    def find_optimal_k(self, max_k: int = 10) -> Tuple[int, float]:
        """
        Find the optimal number of clusters using silhouette score.

        Args:
            max_k: Maximum number of clusters to test

        Returns:
            Tuple of (optimal_k, best_score)
        """
        if not self.keypoints:
            return 1, 0.0

        # Prepare data: flatten all keypoints into a single array
        all_keypoints = []
        for entry in self.keypoints:
            kp_data = entry["keypoints"]["data"]
            if isinstance(kp_data, list) and len(kp_data) > 0:
                # Take only the first person detected in each frame
                first_person = kp_data[0]
                if isinstance(first_person, list) and len(first_person) > 0:
                    # Flatten keypoints (x, y) for the first person
                    flattened = []
                    for kp in first_person:
                        if isinstance(kp, list) and len(kp) >= 2:
                            flattened.extend(kp[:2])  # Only x, y coordinates
                    if flattened:
                        all_keypoints.append(flattened)

        if len(all_keypoints) < 2:
            return 1, 0.0

        # Convert to numpy array
        X = np.array(all_keypoints)

        # Test different k values
        best_k = 2
        best_score = -1

        for k in range(2, min(max_k + 1, len(X))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue

        return best_k, best_score

    def save_clusters(self, filename: str = "clusters.pkl") -> None:
        """
        Save clustering results and KMeans model to files.

        Args:
            filename: Name of the file to save cluster data to (JSON) and model (PKL)
        """
        if not hasattr(self, '_last_clustering_result'):
            print("No clustering results available. Run perform_clustering() first.")
            self.perform_clustering()

        # Save clustering results to JSON
        with open(filename, 'w') as f:
            json.dump(self._last_clustering_result, f, indent=2)

        # Save KMeans model to pickle file
        if hasattr(self, '_kmeans_model'):
            model_filename = filename.replace('.json', '.pkl')
            with open(model_filename, 'wb') as f:
                pickle.dump(self._kmeans_model, f)

        # Print summary
        result = self._last_clustering_result
        print(f"Cluster data saved to {filename}")
        print(f"Summary: {result['n_clusters']} clusters, silhouette score: {result['silhouette_score']:.3f}, {result['total_frames']} frames")

    def load_clusters(self, filename: str = "clusters.pkl") -> Optional[Dict[str, Any]]:
        """
        Load clustering results and KMeans model from files.

        Args:
            filename: Name of the file to load cluster data from (JSON) and model (PKL)

        Returns:
            Dictionary containing the loaded cluster data, or None if file not found
        """
        try:
            # Load clustering results from JSON
            with open(filename, 'r') as f:
                cluster_data = json.load(f)
            self._last_clustering_result = cluster_data

            # Load KMeans model from pickle file
            model_filename = filename.replace('.json', '.pkl')
            try:
                with open(model_filename, 'rb') as f:
                    self._kmeans_model = pickle.load(f)
            except FileNotFoundError:
                # Fallback to reconstructing from centroids if pickle file not found
                print(f"Model file {model_filename} not found, reconstructing from centroids")
                if 'clusters' in cluster_data:
                    centroids = []
                    for cluster_key in sorted(cluster_data['clusters'].keys()):
                        centroids.append(cluster_data['clusters'][cluster_key]['centroid'])
                    if centroids:
                        # Create a dummy KMeans model with the centroids
                        n_clusters = len(centroids)
                        self._kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        self._kmeans_model.cluster_centers_ = np.array(centroids)
                        # Note: This is a simplified reconstruction - full model state isn't saved

            # Print summary
            print(f"Cluster data loaded from {filename}")
            print(f"Summary: {cluster_data['n_clusters']} clusters, silhouette score: {cluster_data['silhouette_score']:.3f}, {cluster_data['total_frames']} frames")
            return cluster_data
        except FileNotFoundError:
            print(f"Cluster file {filename} not found")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing cluster file: {e}")
            return None

    def perform_clustering(self, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform k-means clustering on the collected keypoints.

        Args:
            k: Number of clusters. If None, automatically determine optimal k.

        Returns:
            Dictionary containing clustering results
        """
        result = self._perform_clustering(k)
        self._last_clustering_result = result  # Store for saving
        return result

    def _perform_clustering(self, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Internal method to perform the actual clustering logic.
        """
        if not self.keypoints:
            return {"error": "No keypoints data available"}

        # Prepare data - only use the first detected person per frame
        all_keypoints = []
        timestamps = []
        for entry in self.keypoints:
            kp_data = entry["keypoints"]["data"]
            if isinstance(kp_data, list) and len(kp_data) > 0:
                # Take only the first person detected in each frame
                first_person = kp_data[0]
                if len(first_person) >= 2:
                    # Flatten keypoints (x, y) for the first person
                    flattened = []
                    for kp in first_person:
                        if isinstance(kp, list) and len(kp) >= 2:
                            flattened.extend(kp[:2])  # Only x, y coordinates
                    if flattened:
                        all_keypoints.append(flattened)
                        timestamps.append(entry["timestamp"])

        if len(all_keypoints) < 2:
            return {"error": "Insufficient data for clustering"}

        X = np.array(all_keypoints)

        # Determine k if not provided
        if k is None:
            k, silhouette = self.find_optimal_k()
        else:
            k = min(k, len(X))
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                silhouette = silhouette_score(X, labels)
            except:
                silhouette = 0.0

        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        # Store the trained model for prediction
        self._kmeans_model = kmeans

        # Group data by clusters
        clusters = {}
        for i in range(k):
            cluster_indices = np.where(labels == i)[0]
            cluster_data = {
                "centroid": centroids[i].tolist(),
                "frames": len(cluster_indices),
                "timestamps": [timestamps[idx] for idx in cluster_indices],
                "keypoints": [all_keypoints[idx] for idx in cluster_indices]
            }
            clusters[f"cluster_{i}"] = cluster_data

        return {
            "n_clusters": k,
            "silhouette_score": silhouette,
            "total_frames": len(X),
            "clusters": clusters,
            "inertia": kmeans.inertia_
        }

    def predict_cluster(self, keypoints_data) -> Optional[Dict[str, Any]]:
        """
        Predict which cluster a new keypoints data belongs to and return distance.

        Args:
            keypoints_data: The keypoints data to classify

        Returns:
            Dictionary with 'cluster' (int) and 'distance' (float) or None if no clustering model available
        """
        if not hasattr(self, '_kmeans_model'):
            print("No clustering model available. Run perform_clustering() first.")
            return None

        if keypoints_data is None or keypoints_data.data is None:
            return None
        
        try:
            # Take only the first person detected
            kp_data = keypoints_data.data
            if kp_data is not None and len(kp_data) > 0:
                first_person = kp_data[0]
                if len(first_person) > 0:
                    # Flatten keypoints (x, y) for the first person
                    flattened = []
                    for kp in first_person:
                        if len(kp) >= 2:
                            flattened.extend(kp[:2])  # Only x, y coordinates
                    if flattened:
                        # Convert to numpy array and predict
                        X_new = np.array([flattened])
                        cluster = self._kmeans_model.predict(X_new)[0]

                        # Calculate distance to the assigned cluster centroid
                        centroid = self._kmeans_model.cluster_centers_[cluster]
                        distance = np.linalg.norm(X_new[0] - centroid)

                        return {
                            "cluster": int(cluster),
                            "distance": float(distance)
                        }
            else:
                print(type(kp_data),len(kp_data))

        except Exception as e:
            print(f"Error predicting cluster: {e}")
            raise
            return None

        return None