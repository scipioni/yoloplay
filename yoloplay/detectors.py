"""
Pose detection module with different detector implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Union, Optional

import cv2
import numpy as np


class Keypoints:
    """
    Common keypoints class that normalizes data from YOLO and MediaPipe pose detectors.

    Provides a unified interface for accessing keypoints regardless of the source detector.
    All keypoints are normalized to [0, 1] range for consistent processing.
    """

    # COCO keypoint names for reference
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(
        self,
        keypoints: Union[np.ndarray, Any],
        source: str = "yolo",
        image_shape: Optional[tuple] = None
    ):
        """
        Initialize keypoints from YOLO or MediaPipe data.

        Args:
            keypoints: Raw keypoints data from detector
            source: Source detector ("yolo" or "mediapipe")
            image_shape: Image dimensions (height, width) for YOLO normalization
        """
        self.source = source.lower()
        self.image_shape = image_shape
        self._keypoints = None
        self._original_landmarks = None  # Store original MediaPipe landmarks

        if self.source == "yolo":
            self._normalize_yolo_keypoints(keypoints)
        elif self.source == "mediapipe":
            self._original_landmarks = keypoints  # Store original landmarks
            self._normalize_mediapipe_keypoints(keypoints)
        else:
            raise ValueError(f"Unsupported source: {source}. Must be 'yolo' or 'mediapipe'")

    def _normalize_yolo_keypoints(self, keypoints: np.ndarray) -> None:
        """
        Normalize YOLO keypoints to [0, 1] range.

        Args:
            keypoints: YOLO keypoints array (num_persons, 17, 3) - x, y, conf
        """
        if self.image_shape is None:
            raise ValueError("image_shape must be provided for YOLO keypoints normalization")

        height, width = self.image_shape

        # Handle tensor conversion if needed
        if hasattr(keypoints, 'cpu'):
            keypoints = keypoints.cpu().numpy()

        # Normalize x, y coordinates to [0, 1] for all persons
        normalized_keypoints = keypoints.copy()
        normalized_keypoints[:, :, 0] /= width   # x coordinates
        normalized_keypoints[:, :, 1] /= height  # y coordinates

        self._keypoints = normalized_keypoints

    def _normalize_mediapipe_keypoints(self, landmarks: Any) -> None:
        """
        Extract and store MediaPipe keypoints (already normalized).

        Args:
            landmarks: MediaPipe pose landmarks object
        """
        if not landmarks:
            self._keypoints = np.zeros((33, 3))  # MediaPipe has 33 landmarks
            return

        # MediaPipe landmarks are already normalized to [0, 1]
        keypoints = np.zeros((33, 3))
        for i, landmark in enumerate(landmarks.landmark):
            keypoints[i] = [landmark.x, landmark.y, landmark.visibility]

        self._keypoints = keypoints

    @property
    def data(self) -> np.ndarray:
        """Get the normalized keypoints array."""
        return self._keypoints

    @property
    def original_landmarks(self) -> Any:
        """Get the original MediaPipe landmarks object."""
        return self._original_landmarks

    @property
    def num_persons(self) -> int:
        """Get the number of persons detected."""
        if self._keypoints is None:
            return 0
        if self.source == "yolo":
            return self._keypoints.shape[0] if len(self._keypoints.shape) == 3 else 1
        else:  # mediapipe
            return 1  # MediaPipe only detects one person

    def get_person_keypoints(self, person_idx: int = 0) -> np.ndarray:
        """
        Get keypoints for a specific person.

        Args:
            person_idx: Index of the person (0-based)

        Returns:
            Keypoints array for the specified person
        """
        if self._keypoints is None:
            return np.zeros((17 if self.source == "yolo" else 33, 3))

        if self.source == "yolo":
            if len(self._keypoints.shape) == 3 and person_idx < self._keypoints.shape[0]:
                return self._keypoints[person_idx]
            elif person_idx == 0:
                return self._keypoints
            else:
                return np.zeros((17, 3))
        else:  # mediapipe
            return self._keypoints if person_idx == 0 else np.zeros((33, 3))

    def get_keypoint(self, index: int) -> np.ndarray:
        """
        Get a specific keypoint by index.

        Args:
            index: Keypoint index

        Returns:
            Array [x, y, confidence] for the keypoint
        """
        if self._keypoints is None or index >= len(self._keypoints):
            return np.array([0.0, 0.0, 0.0])
        return self._keypoints[index]

    def get_keypoint_by_name(self, name: str) -> np.ndarray:
        """
        Get a keypoint by name (COCO keypoints only).

        Args:
            name: Keypoint name (e.g., "nose", "left_shoulder")

        Returns:
            Array [x, y, confidence] for the keypoint
        """
        if self.source == "mediapipe":
            # Map COCO names to MediaPipe indices for common keypoints
            coco_to_mediapipe = {
                "nose": 0,
                "left_eye": 1,
                "right_eye": 2,
                "left_ear": 7,
                "right_ear": 8,
                "left_shoulder": 11,
                "right_shoulder": 12,
                "left_elbow": 13,
                "right_elbow": 14,
                "left_wrist": 15,
                "right_wrist": 16,
                "left_hip": 23,
                "right_hip": 24,
                "left_knee": 25,
                "right_knee": 26,
                "left_ankle": 27,
                "right_ankle": 28,
            }
            if name in coco_to_mediapipe:
                return self.get_keypoint(coco_to_mediapipe[name])
        else:
            # YOLO uses COCO format
            if name in self.COCO_KEYPOINTS:
                index = self.COCO_KEYPOINTS.index(name)
                return self.get_keypoint(index)

        return np.array([0.0, 0.0, 0.0])

    def print_keypoints(self) -> None:
        """
        Print keypoints in a readable format.
        """
        if self._keypoints is None:
            print("No keypoints available")
            return

        print(f"Keypoints from {self.source.upper()} detector:")
        print(f"Total keypoints: {len(self._keypoints)}")
        print("-" * 50)

        if self.source == "yolo":
            # Print COCO keypoints
            for i, name in enumerate(self.COCO_KEYPOINTS):
                if i < len(self._keypoints):
                    x, y, conf = self._keypoints[i]
                    print("2d")
        elif self.source == "mediapipe":
            # Print MediaPipe landmark names (simplified)
            landmark_names = [
                "nose", "left_eye_inner", "left_eye", "left_eye_outer",
                "right_eye_inner", "right_eye", "right_eye_outer",
                "left_ear", "right_ear", "mouth_left", "mouth_right",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_pinky", "right_pinky",
                "left_index", "right_index", "left_thumb", "right_thumb",
                "left_hip", "right_hip", "left_knee", "right_knee",
                "left_ankle", "right_ankle", "left_heel", "right_heel",
                "left_foot_index", "right_foot_index"
            ]

            for i, (x, y, conf) in enumerate(self._keypoints):
                name = landmark_names[i] if i < len(landmark_names) else f"landmark_{i}"
                print("2d")

        print("-" * 50)


class PoseDetector(ABC):
    """Abstract base class for pose detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Keypoints:
        """
        Detect pose in the given frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Keypoints object with normalized pose data
        """
        pass

    @abstractmethod
    def visualize(self, frame: np.ndarray, keypoints: Keypoints) -> np.ndarray:
        """
        Visualize detection results on the frame.

        Args:
            frame: Input frame (BGR format)
            keypoints: Keypoints object from detect()

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

    def detect(self, frame: np.ndarray) -> Keypoints:
        """
        Detect pose using YOLO model.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Keypoints object with normalized pose data for all detected persons
        """
        results = self.model(frame)

        # Extract keypoints from YOLO results
        keypoints_data = None
        for r in results:
            if hasattr(r, "keypoints") and r.keypoints is not None:
                keypoints_data = r.keypoints.data
                break

        if keypoints_data is None:
            # Return empty keypoints if no detection
            return Keypoints(np.zeros((0, 17, 3)), source="yolo", image_shape=frame.shape[:2])

        return Keypoints(keypoints_data, source="yolo", image_shape=frame.shape[:2])

    def visualize(self, frame: np.ndarray, keypoints: Keypoints) -> np.ndarray:
        """
        Visualize YOLO pose detection results for all persons.

        Args:
            frame: Input frame (BGR format)
            keypoints: Keypoints object from detect()

        Returns:
            Annotated frame with pose keypoints and skeleton for all persons
        """
        annotated_frame = frame.copy()

        # Get keypoints data for visualization
        keypoints_data = keypoints.data

        # Draw skeleton for each person detected
        if keypoints_data is not None and len(keypoints_data) > 0:
            num_persons = keypoints.num_persons

            for person_idx in range(num_persons):
                person_kpts = keypoints.get_person_keypoints(person_idx)

                # Draw connections (bones) between keypoints for this person
                for i, sk in enumerate(self.skeleton):
                    # Convert from 1-based to 0-based indexing and check bounds
                    idx1 = sk[0] - 1
                    idx2 = sk[1] - 1

                    if idx1 >= len(person_kpts) or idx2 >= len(person_kpts):
                        continue

                    # Convert normalized coordinates back to pixel coordinates
                    pos1 = (int(person_kpts[idx1][0] * frame.shape[1]), int(person_kpts[idx1][1] * frame.shape[0]))
                    pos2 = (int(person_kpts[idx2][0] * frame.shape[1]), int(person_kpts[idx2][1] * frame.shape[0]))

                    # Check if both points have high confidence
                    conf1 = person_kpts[idx1][2]
                    conf2 = person_kpts[idx2][2]

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

    def detect(self, frame: np.ndarray) -> Keypoints:
        """
        Detect pose using MediaPipe.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Keypoints object with normalized pose data
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            return Keypoints(results.pose_landmarks, source="mediapipe")
        else:
            # Return empty keypoints if no detection
            return Keypoints(None, source="mediapipe")

    def visualize(self, frame: np.ndarray, keypoints: Keypoints, fall_detected: bool = False) -> np.ndarray:
        """
        Visualize MediaPipe pose detection results.

        Args:
            frame: Input frame (BGR format)
            keypoints: Keypoints object from detect()
            fall_detected: Whether a fall was detected (affects visualization colors)

        Returns:
            Annotated frame with pose landmarks and connections
        """
        # If no pose was detected, return the original frame without visualization
        if keypoints.original_landmarks is None:
            return frame.copy()

        annotated_frame = frame.copy()

        # Get keypoints data (already normalized)
        keypoints_data = keypoints.data

        if keypoints_data is not None and len(keypoints_data) > 0:
            # Calculate bounding box from keypoints
            x_coords = keypoints_data[:, 0]
            y_coords = keypoints_data[:, 1]

            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)

            # Draw bounding box with red color if fall detected
            if fall_detected:
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
            else:
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

            # Draw pose landmarks and connections with default colors
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                keypoints.original_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2
                ),
            )

        return annotated_frame