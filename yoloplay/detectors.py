"""
Pose detection module with different detector implementations.
"""

from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np


class PoseDetector(ABC):
    """Abstract base class for pose detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Any:
        """
        Detect pose in the given frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Detection results (format depends on implementation)
        """
        pass

    @abstractmethod
    def visualize(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Visualize detection results on the frame.

        Args:
            frame: Input frame (BGR format)
            results: Detection results from detect()

        Returns:
            Annotated frame with pose visualization
        """
        pass


class YOLOPoseDetector(PoseDetector):
    """YOLO-based pose detector implementation."""

    # YOLO skeleton connections for drawing bones
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # legs and center
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],  # body and arms
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],  # arms, face
        [2, 4], [3, 5], [4, 6], [5, 7],  # face to shoulders to arms
    ]

    # Colors for different body parts
    pose_palette = np.array(
        [
            [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],  # legs
            [255, 51, 51], [255, 51, 51], [255, 51, 51], [255, 102, 66], [255, 102, 66],  # body and arms
            [255, 102, 66], [255, 102, 66], [51, 153, 51], [51, 153, 51], [51, 153, 51],  # arms and face
            [51, 153, 51], [51, 153, 51], [51, 153, 51], [51, 153, 51],  # face to shoulders
        ],
        dtype=np.uint8,
    ).tolist()

    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        """
        Initialize YOLO pose detector.

        Args:
            model_path: Path to YOLO pose model weights
        """
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> Any:
        """
        Detect pose using YOLO model.

        Args:
            frame: Input frame (BGR format)

        Returns:
            YOLO results object
        """
        return self.model(frame)

    def visualize(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Visualize YOLO pose detection results.

        Args:
            frame: Input frame (BGR format)
            results: YOLO results object

        Returns:
            Annotated frame with pose keypoints and skeleton
        """
        annotated_frame = frame.copy()

        # Process results
        for r in results:
            # Plot boxes and poses on the frame
            annotated_frame = r.plot()

            # Draw skeleton if keypoints are available
            if hasattr(r, "keypoints") and r.keypoints is not None:
                keypoints = r.keypoints.data  # Shape: (num_persons, num_keypoints, 3)

                # Iterate through each person detected
                for person_kpts in keypoints:
                    if person_kpts is not None:
                        # Draw connections (bones) between keypoints
                        for i, sk in enumerate(self.skeleton):
                            pos1 = (int(person_kpts[sk[0] - 1][0]), int(person_kpts[sk[0] - 1][1]))
                            pos2 = (int(person_kpts[sk[1] - 1][0]), int(person_kpts[sk[1] - 1][1]))

                            # Check if both points have high confidence
                            conf1 = person_kpts[sk[0] - 1][2]
                            conf2 = person_kpts[sk[1] - 1][2]

                            if (conf1 > 0.5 and conf2 > 0.5 and 
                                pos1[0] > 0 and pos1[1] > 0 and 
                                pos2[0] > 0 and pos2[1] > 0):
                                # Draw the bone (line between keypoints)
                                color = self.pose_palette[i]
                                cv2.line(annotated_frame, pos1, pos2, color, 
                                        thickness=2, lineType=cv2.LINE_AA)

        return annotated_frame


class MediaPipePoseDetector(PoseDetector):
    """MediaPipe-based pose detector implementation."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe pose detector.

        Args:
            static_image_mode: Whether to treat input as static images
            model_complexity: Model complexity (0, 1, or 2)
            enable_segmentation: Whether to enable segmentation
            min_detection_confidence: Minimum detection confidence threshold
        """
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame: np.ndarray) -> Any:
        """
        Detect pose using MediaPipe.

        Args:
            frame: Input frame (BGR format)

        Returns:
            MediaPipe pose results object
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)

    def visualize(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Visualize MediaPipe pose detection results.

        Args:
            frame: Input frame (BGR format)
            results: MediaPipe pose results object

        Returns:
            Annotated frame with pose landmarks and connections
        """
        annotated_frame = frame.copy()

        if results.pose_landmarks:
            # Draw pose landmarks and connections
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2
                ),
            )

        return annotated_frame