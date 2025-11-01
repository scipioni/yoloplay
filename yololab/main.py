import cv2
import numpy as np
from typing import Tuple, Optional, Generator
from ultralytics import YOLO


class CameraPoseProcessor:
    """
    Class to handle camera input, YOLO Pose detection, and coordinate transformation
    """

    def __init__(self, config_path: str, model_path: str = "yolov8n-pose.pt"):
        self.config_path = (
            config_path  # Store config path for saving calibration points
        )

        # Load YOLO Pose model
        self.model = YOLO(model_path)

    def run_camera_loop(
        self, camera_index: int = 0, display_original: bool = True
    ) -> None:
        """
        Run the main camera processing loop

        Args:
            camera_index: Index of the camera to use (default: 0)
            display_original: Whether to display both original and transformed frames
        """
        # Open camera
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise ValueError(f"Cannot open camera with index {camera_index}")

        print(f"Camera opened successfully. Press 'q' to quit.")

        while True:
            # Read frame from camera
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame from camera")
                break

            cv2.imshow("camera", frame)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release camera and close windows
        cap.release()
        cv2.destroyAllWindows()


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

    args = parser.parse_args()

    processor = CameraPoseProcessor(args.config, args.model)

    processor.run_camera_loop(args.camera)


if __name__ == "__main__":
    main()
