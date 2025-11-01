import os
import time
from typing import Generator, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .utils import draw_pose_estimation


class CameraPoseProcessor:
    """
    Class to handle camera input, YOLO Pose detection, and coordinate transformation
    """

    def __init__(self, config_path: str, model_path: str = "yolov8n-pose.pt", camera_height: float = 130.0):
        self.config_path = (
            config_path  # Store config path for saving calibration points
        )

        # Load YOLO Pose model
        self.model = YOLO(model_path)
        
        # Store camera height in cm
        self.camera_height = camera_height

    def run_camera_loop(
        self, camera_index: int = 0, display_original: bool = True
    ) -> None:
        """
        Run the main camera processing loop

        Args:
            camera_index: Index of the camera to use (default: 0)
            display_original: Whether to display both original and transformed frames
        """
        # Check if display is available (for headless environments)
        display_available = self._check_display_available()

        # Open camera
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise ValueError(f"Cannot open camera with index {camera_index}")

        if display_available:
            print(f"Camera opened successfully at height {self.camera_height}cm. Press 'q' to quit.")
        else:
            print(f"Camera opened successfully in headless mode at height {self.camera_height}cm.")

        while True:
            # Read frame from camera
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame from camera")
                break

            # Only show the frame if display is available
            if display_available:
                # Run pose detection
                results = self.model(frame)

                # Draw pose estimation with bones on the frame
                annotated_frame = draw_pose_estimation(frame, results)

                # Show the annotated frame
                cv2.imshow("camera", annotated_frame)

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # In headless mode, just print a message every few frames to show activity

                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                # Break on Ctrl+C
                try:
                    # Do nothing for a while, check periodically if we need to quit
                    pass
                except KeyboardInterrupt:
                    print("Interrupted by user")
                    break

        # Release camera and close windows if display is available
        cap.release()
        if display_available:
            cv2.destroyAllWindows()

    def _check_display_available(self) -> bool:
        """
        Check if a display is available (for GUI operations)
        """
        # Check if running in a container without display
        if os.environ.get("DISPLAY"):
            return True
        # On Linux, try to access X11 display
        try:
            import subprocess

            result = subprocess.run(
                ["xdpyinfo"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return result.returncode == 0
        except FileNotFoundError:
            # xdpyinfo not available, assume no display
            return False
        except:
            # Other error, safer to assume no display
            return False


def main():
    """Command line entry point for camera functionality"""
    import argparse

    parser = argparse.ArgumentParser(description="yoloplay")
    parser.add_argument(
        "--config", type=str, default="./config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-pose.pt",
        help="YOLO Pose model path (default: yolov8n-pose.pt)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=130.0,
        help="Camera height in cm (default: 130.0)",
    )

    args = parser.parse_args()

    processor = CameraPoseProcessor(args.config, args.model, args.height)

    processor.run_camera_loop(args.camera)


if __name__ == "__main__":
    main()
