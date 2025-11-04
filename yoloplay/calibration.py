"""
Calibration module for collecting keypoints during pose detection.
"""

from typing import List, Dict, Any
import json
import time


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

    def save_to_file(self, filename: str = "calibration_data.json") -> None:
        """
        Save collected keypoints to a JSON file.

        Args:
            filename: Name of the file to save to
        """
        with open(filename, 'w') as f:
            json.dump(self.keypoints, f, indent=2)
        print(f"Calibration data saved to {filename}")

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