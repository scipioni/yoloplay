import os
import time
from typing import Generator, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .utils import draw_pose_estimation, draw_mediapipe_pose_estimation


class CameraPoseProcessor:
    """
    Class to handle camera input, YOLO Pose detection, and coordinate transformation
    """

    def __init__(self, config_path: str, model_path: str = "yolov8n-pose.pt", camera_height: float = 130.0, use_mediapipe: bool = False):
        self.config_path = (
            config_path  # Store config path for saving calibration points
        )

        self.use_mediapipe = use_mediapipe

        if use_mediapipe:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        else:
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
                if self.use_mediapipe:
                    # Convert BGR to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Run MediaPipe pose detection
                    pose_results = self.pose.process(rgb_frame)
                    # Draw pose estimation on the frame
                    annotated_frame = draw_mediapipe_pose_estimation(frame, pose_results)
                else:
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

    def run_video_loop(self, video_path: str, display_original: bool = True) -> None:
        """
        Run the video processing loop

        Args:
            video_path: Path to the video file
            display_original: Whether to display the processed frames
        """
        # Check if display is available (for headless environments)
        display_available = self._check_display_available()

        # Open video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file {video_path}")

        if display_available:
            print(f"Video opened successfully. Press 'q' to quit.")
        else:
            print(f"Video opened successfully in headless mode.")

        while True:
            # Read frame from video
            ret, frame = cap.read()

            if not ret:
                print("End of video or failed to grab frame")
                break

            # Only show the frame if display is available
            if display_available:
                if self.use_mediapipe:
                    # Convert BGR to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Run MediaPipe pose detection
                    pose_results = self.pose.process(rgb_frame)
                    # Draw pose estimation on the frame
                    annotated_frame = draw_mediapipe_pose_estimation(frame, pose_results)
                else:
                    # Run pose detection
                    results = self.model(frame)
                    # Draw pose estimation with bones on the frame
                    annotated_frame = draw_pose_estimation(frame, results)

                # Show the annotated frame
                cv2.imshow("video", annotated_frame)

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

        # Release video and close windows if display is available
        cap.release()
        if display_available:
            cv2.destroyAllWindows()

    def run_images_loop(self, image_paths: list[str], display_original: bool = True) -> None:
        """
        Run the images processing loop

        Args:
            image_paths: List of paths to image files
            display_original: Whether to display the processed images
        """
        # Check if display is available (for headless environments)
        display_available = self._check_display_available()

        for image_path in image_paths:
            # Read image
            frame = cv2.imread(image_path)

            if frame is None:
                print(f"Failed to load image {image_path}")
                continue

            if self.use_mediapipe:
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Run MediaPipe pose detection
                pose_results = self.pose.process(rgb_frame)
                # Draw pose estimation on the frame
                annotated_frame = draw_mediapipe_pose_estimation(frame, pose_results)
            else:
                # Run pose detection
                results = self.model(frame)
                # Draw pose estimation with bones on the frame
                annotated_frame = draw_pose_estimation(frame, results)

            if display_available:
                # Show the annotated frame
                cv2.imshow("image", annotated_frame)
                cv2.waitKey(0)  # Wait for key press to show next image
            else:
                # In headless mode, just print a message
                print(f"Processed image {image_path}")

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
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file to process",
    )
    parser.add_argument(
        "--images",
        nargs='+',
        help="List of image files to process",
    )
    parser.add_argument(
        "--mediapipe",
        action="store_true",
        help="Use MediaPipe for pose detection instead of YOLO",
    )

    args = parser.parse_args()

    processor = CameraPoseProcessor(args.config, args.model, args.height, args.mediapipe)

    if args.video:
        processor.run_video_loop(args.video)
    elif args.images:
        processor.run_images_loop(args.images)
    else:
        processor.run_camera_loop(args.camera)


if __name__ == "__main__":
    main()
