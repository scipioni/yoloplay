from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import cv2
import numpy as np

from .autoencoder import OneClassAutoencoderClassifier


class Keypoint:
    """
    Represents a single person's pose keypoints.

    Stores normalized keypoints data with additional metadata like bounding box
    and person ID. Provides methods for accessing and filtering keypoints.
    """

    # COCO keypoint names for reference
    COCO_KEYPOINTS = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(
        self,
        keypoints: Union[np.ndarray, Any],
        source: str = "yolo",
        person_id: Optional[int] = None,
        image_shape: Optional[tuple] = None,
        normalize: bool = False,
    ):
        """
        Initialize a single person's keypoints.

        Args:
            keypoints: Keypoints array or MediaPipe landmarks object
            source: Source detector ("yolo" or "mediapipe")
            person_id: Optional person identifier
            image_shape: Optional image dimensions (height, width)
            normalize: Whether to normalize the keypoints
        """
        self.source = source.lower()
        self.person_id = person_id
        self.image_shape = image_shape
        self._bbox = None  # Lazy computed bounding box
        self.anomaly_detected: bool = False
        self.anomaly_score: float = 0.0
        self.anomaly_method: str = ""  # Track which method detected the anomaly

        if normalize:
            if self.source == "yolo":
                self._data = self._normalize_yolo(keypoints, image_shape)
            elif self.source == "mediapipe":
                self._data = self._normalize_mediapipe(keypoints)
            else:
                raise ValueError(f"Unsupported source: {source}")
        else:
            self._data = keypoints

    @staticmethod
    def _normalize_yolo(keypoints: np.ndarray, image_shape: tuple) -> np.ndarray:
        """
        Normalize YOLO keypoints from pixel to [0, 1] range.

        Args:
            keypoints: YOLO keypoints array (17, 3) - x, y, conf in pixels
            image_shape: Image dimensions (height, width)

        Returns:
            Normalized keypoints array (17, 3)
        """
        if image_shape is None:
            raise ValueError(
                "image_shape must be provided for YOLO keypoints normalization"
            )

        height, width = image_shape

        # Handle tensor conversion if needed
        if hasattr(keypoints, "cpu"):
            keypoints = keypoints.cpu().numpy()

        # Normalize x, y coordinates to [0, 1]
        normalized = keypoints.copy()
        normalized[:, 0] /= width  # x coordinates
        normalized[:, 1] /= height  # y coordinates

        return normalized

    @staticmethod
    def _normalize_mediapipe(landmarks: Any) -> np.ndarray:
        """
        Extract and normalize MediaPipe keypoints.

        Args:
            landmarks: MediaPipe pose landmarks object

        Returns:
            Normalized keypoints array (33, 3)
        """
        if not landmarks:
            return np.zeros((33, 3))

        # MediaPipe landmarks are already normalized to [0, 1]
        keypoints = np.zeros((33, 3))
        for i, landmark in enumerate(landmarks.landmark):
            keypoints[i] = [landmark.x, landmark.y, landmark.visibility]

        return keypoints

    @property
    def data(self) -> np.ndarray:
        """Get the normalized keypoints array."""
        return self._data

    @property
    def xy(self) -> np.ndarray:
        """Get the normalized keypoints array."""
        return self._data[:, :2]

    @property
    def confidence(self) -> float:
        """Get the average confidence/visibility for this person."""
        if self._data is None or len(self._data) == 0:
            return 0.0
        return np.mean(self._data[:, 2])

    @property
    def bbox(self) -> Optional[tuple]:
        """
        Get the bounding box (x_min, y_min, x_max, y_max) in normalized coordinates.
        Lazy computed and cached.
        """
        if self._bbox is None and self._data is not None:
            # Filter high confidence keypoints
            valid_kpts = self._data[self._data[:, 2] > 0.5]
            if len(valid_kpts) > 0:
                x_coords = valid_kpts[:, 0]
                y_coords = valid_kpts[:, 1]
                self._bbox = (
                    float(min(x_coords)),
                    float(min(y_coords)),
                    float(max(x_coords)),
                    float(max(y_coords)),
                )
        return self._bbox

    def get_bbox_pixels(self) -> Optional[tuple]:
        """
        Get the bounding box in pixel coordinates.
        Requires image_shape to be set.

        Returns:
            Tuple (x_min, y_min, x_max, y_max) in pixels or None
        """
        if self.bbox is None or self.image_shape is None:
            return None

        h, w = self.image_shape
        x_min, y_min, x_max, y_max = self.bbox
        return (int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h))

    def get_keypoint(self, index: int) -> np.ndarray:
        """
        Get a specific keypoint by index.

        Args:
            index: Keypoint index

        Returns:
            Array [x, y, confidence] for the keypoint
        """
        if self._data is None or index >= len(self._data):
            return np.array([0.0, 0.0, 0.0])
        return self._data[index]

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

    def has_visible_arm(self, min_conf: float = 0.5) -> bool:
        """Check if person has at least one visible arm."""
        if self.source == "yolo":
            left_arm_indices = [5, 7, 9]  # left_shoulder, left_elbow, left_wrist
            right_arm_indices = [6, 8, 10]  # right_shoulder, right_elbow, right_wrist
        else:  # mediapipe
            left_arm_indices = [11, 13, 15]
            right_arm_indices = [12, 14, 16]

        left_arm = any(self._data[i, 2] >= min_conf for i in left_arm_indices)
        right_arm = any(self._data[i, 2] >= min_conf for i in right_arm_indices)

        return left_arm or right_arm

    def has_visible_leg(self, min_conf: float = 0.5) -> bool:
        """Check if person has at least one visible leg."""
        if self.source == "yolo":
            left_leg_indices = [11, 13, 15]  # left_hip, left_knee, left_ankle
            right_leg_indices = [12, 14, 16]  # right_hip, right_knee, right_ankle
        else:  # mediapipe
            left_leg_indices = [23, 25, 27]
            right_leg_indices = [24, 26, 28]

        left_leg = any(self._data[i, 2] >= min_conf for i in left_leg_indices)
        right_leg = any(self._data[i, 2] >= min_conf for i in right_leg_indices)

        return left_leg or right_leg

    def get_kpts_xy(self) -> list:
        """
        Get keypoints as flattened list of x, y coordinates (without confidence).

        Returns:
            List of [x1, y1, x2, y2, ..., xn, yn]
        """
        if self._data is None:
            return []
        result = []
        for point in self._data:
            result.append(point[0])
            result.append(point[1])
        return result

    def print_keypoints(self) -> None:
        """Print this person's keypoints in a readable format."""
        if self._data is None or len(self._data) == 0:
            print("No keypoints available")
            return

        print(f"Person ID: {self.person_id}, Confidence: {self.confidence:.3f}")

        if self.source == "yolo":
            # Print COCO keypoints
            for i, name in enumerate(self.COCO_KEYPOINTS):
                if i < len(self._data):
                    x, y, conf = self._data[i]
                    print(f"  {name:20s}: x={x:.3f}, y={y:.3f}, conf={conf:.3f}")
        elif self.source == "mediapipe":
            # Print MediaPipe landmark names
            landmark_names = [
                "nose",
                "left_eye_inner",
                "left_eye",
                "left_eye_outer",
                "right_eye_inner",
                "right_eye",
                "right_eye_outer",
                "left_ear",
                "right_ear",
                "mouth_left",
                "mouth_right",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_pinky",
                "right_pinky",
                "left_index",
                "right_index",
                "left_thumb",
                "right_thumb",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "left_heel",
                "right_heel",
                "left_foot_index",
                "right_foot_index",
            ]

            for i, (x, y, conf) in enumerate(self._data):
                name = landmark_names[i] if i < len(landmark_names) else f"landmark_{i}"
                print(f"  {name:20s}: x={x:.3f}, y={y:.3f}, conf={conf:.3f}")

        # Print bounding box if available
        bbox = self.bbox
        if bbox:
            print(
                f"  Bounding box: x_min={bbox[0]:.3f}, y_min={bbox[1]:.3f}, x_max={bbox[2]:.3f}, y_max={bbox[3]:.3f}"
            )
    
    def draw_skeleton(self, frame: np.ndarray, skeleton: list, pose_palette: list) -> np.ndarray:
        """
        Draw skeleton connections on frame for YOLO-style visualization.
        
        Args:
            frame: Frame to draw on
            skeleton: List of keypoint connection pairs
            pose_palette: Color palette for connections
            
        Returns:
            Frame with skeleton drawn
        """
        if self._data is None or len(self._data) == 0:
            return frame
        
        for i, sk in enumerate(skeleton):
            # Convert from 1-based to 0-based indexing
            idx1 = sk[0] - 1
            idx2 = sk[1] - 1
            
            if idx1 >= len(self._data) or idx2 >= len(self._data):
                continue
            
            # Convert normalized coordinates to pixel coordinates
            pos1 = (
                int(self._data[idx1][0] * frame.shape[1]),
                int(self._data[idx1][1] * frame.shape[0]),
            )
            pos2 = (
                int(self._data[idx2][0] * frame.shape[1]),
                int(self._data[idx2][1] * frame.shape[0]),
            )
            
            # Check if both points have high confidence
            conf1 = self._data[idx1][2]
            conf2 = self._data[idx2][2]
            
            if (
                conf1 > 0.5
                and conf2 > 0.5
                and pos1[0] > 0
                and pos1[1] > 0
                and pos2[0] > 0
                and pos2[1] > 0
            ):
                # Draw the bone
                color = pose_palette[i]
                cv2.line(frame, pos1, pos2, color, thickness=2, lineType=cv2.LINE_AA)
        
        return frame
    
    def draw_bbox(self, frame: np.ndarray, color: tuple = (128, 128, 128), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box on frame.
        
        Args:
            frame: Frame to draw on
            color: BGR color tuple for the box
            thickness: Line thickness
            
        Returns:
            Frame with bounding box drawn
        """
        if self._data is None or len(self._data) == 0:
            return frame
        
        # Get high confidence keypoints
        valid_kpts = self._data[self._data[:, 2] > 0.5]
        if len(valid_kpts) == 0:
            return frame
        
        x_coords = valid_kpts[:, 0]
        y_coords = valid_kpts[:, 1]
        
        # Convert normalized to pixel coordinates
        h, w = frame.shape[:2]
        x_min = int(min(x_coords) * w)
        x_max = int(max(x_coords) * w)
        y_min = int(min(y_coords) * h)
        y_max = int(max(y_coords) * h)
        
        # Draw bounding box
        if self.anomaly_detected:
            color = (0, 0, 255)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

        # Add confidence text
        text = f"{self.confidence:.2f}"
        if self.anomaly_method:
            text += f" ({self.anomaly_method})"
        cv2.putText(
            frame,
            text,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        
        return frame
    
    def draw_mediapipe_landmarks(
        self,
        frame: np.ndarray,
        mp_drawing,
        mp_pose,
        original_landmarks: Any,
        fall_detected: bool = False
    ) -> np.ndarray:
        """
        Draw MediaPipe landmarks and connections on frame.
        
        Args:
            frame: Frame to draw on
            mp_drawing: MediaPipe drawing utilities
            mp_pose: MediaPipe pose module
            original_landmarks: Original MediaPipe landmarks object
            fall_detected: Whether a fall was detected (affects box color)
            
        Returns:
            Frame with landmarks and bounding box drawn
        """
        if self._data is None or len(self._data) == 0:
            return frame
        
        # Draw bounding box with appropriate color
        bbox_color = (0, 0, 255) if fall_detected else (0, 255, 0)
        frame = self.draw_bbox(frame, color=bbox_color, thickness=3)
        
        # Draw pose landmarks and connections
        if original_landmarks is not None:
            mp_drawing.draw_landmarks(
                frame,
                original_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2
                ),
            )
        
        return frame


class Keypoints:
    """
    Common keypoints class that normalizes data from YOLO and MediaPipe pose detectors.

    Provides a unified interface for accessing keypoints regardless of the source detector.
    All keypoints are normalized to [0, 1] range for consistent processing.
    """

    def __init__(
        self,
        keypoints: Union[np.ndarray, Any],
        source: str = "yolo",
        image_shape: Optional[tuple] = None,
        normalize=True,
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
        self._original_landmarks = None  # Store original MediaPipe landmarks
        self._keypoints = []  # List of Keypoint objects

        if normalize:
            if self.source == "yolo":
                self._normalize_yolo_keypoints(keypoints)
            elif self.source == "mediapipe":
                self._original_landmarks = keypoints  # Store original landmarks
                self._normalize_mediapipe_keypoints(keypoints)
            else:
                raise ValueError(
                    f"Unsupported source: {source}. Must be 'yolo' or 'mediapipe'"
                )
        else:
            # When normalize=False, convert raw arrays to Keypoint objects
            self._create_keypoints_from_array(keypoints)

    def _normalize_yolo_keypoints(self, keypoints: np.ndarray) -> None:
        """
        Normalize YOLO keypoints to [0, 1] range and create Keypoint objects.

        Args:
            keypoints: YOLO keypoints array (num_persons, 17, 3) - x, y, conf
        """
        # Handle tensor conversion if needed
        if hasattr(keypoints, "cpu"):
            keypoints = keypoints.cpu().numpy()

        # Create Keypoint objects for each person using Keypoint's normalization
        self._keypoints = []
        if len(keypoints.shape) == 3:  # Multiple persons
            for person_idx, person_kpts in enumerate(keypoints):
                kp = Keypoint(
                    person_kpts,
                    source=self.source,
                    person_id=person_idx,
                    image_shape=self.image_shape,
                    normalize=True,
                )
                self._keypoints.append(kp)
        elif len(keypoints.shape) == 2:  # Single person
            kp = Keypoint(
                keypoints,
                source=self.source,
                person_id=0,
                image_shape=self.image_shape,
                normalize=True,
            )
            self._keypoints.append(kp)

    def _normalize_mediapipe_keypoints(self, landmarks: Any) -> None:
        """
        Extract and store MediaPipe keypoints (already normalized) as Keypoint objects.

        Args:
            landmarks: MediaPipe pose landmarks object
        """
        self._keypoints = []

        # Create Keypoint object using Keypoint's normalization
        kp = Keypoint(
            landmarks,
            source=self.source,
            person_id=0,
            image_shape=self.image_shape,
            normalize=True,
        )
        self._keypoints.append(kp)

    def _create_keypoints_from_array(self, keypoints_array: np.ndarray) -> None:
        """
        Create Keypoint objects from already normalized array.

        Args:
            keypoints_array: Normalized keypoints array
        """
        self._keypoints = []

        if keypoints_array is None:
            return

        if len(keypoints_array.shape) == 3:  # Multiple persons (YOLO)
            for person_idx, person_kpts in enumerate(keypoints_array):
                kp = Keypoint(
                    person_kpts,
                    source=self.source,
                    person_id=person_idx,
                    image_shape=self.image_shape,
                )
                self._keypoints.append(kp)
        elif len(keypoints_array.shape) == 2:  # Single person
            kp = Keypoint(
                keypoints_array,
                source=self.source,
                person_id=0,
                image_shape=self.image_shape,
            )
            self._keypoints.append(kp)

    def __iter__(self):
        """Allow iteration over Keypoint objects."""
        return iter(self._keypoints)
    
    def __len__(self) -> int:
        """Return the number of persons detected."""
        return len(self._keypoints)
    
    def __getitem__(self, index: int) -> Keypoint:
        """
        Allow indexing to access Keypoint objects.
        
        Args:
            index: Index of the person
            
        Returns:
            Keypoint object at the specified index
        """
        return self._keypoints[index]

    @property
    def data(self) -> np.ndarray:
        """Get the normalized keypoints array for backward compatibility."""
        if not self._keypoints:
            return None

        # For single person, return just the array
        if len(self._keypoints) == 1:
            return self._keypoints[0].data

        # For multiple persons, stack their data
        return np.array([kp.data for kp in self._keypoints])

    @property
    def original_landmarks(self) -> Any:
        """Get the original MediaPipe landmarks object."""
        return self._original_landmarks

    @property
    def person_confidence(self) -> float:
        """Get the average confidence/visibility for the first person."""
        if not self._keypoints:
            return 0.0
        return self._keypoints[0].confidence

    @property
    def num_persons(self) -> int:
        """Get the number of persons detected."""
        return len(self._keypoints)

    def get_person_keypoints(self, person_idx: int = 0) -> np.ndarray:
        """
        Get keypoints for a specific person.

        Args:
            person_idx: Index of the person (0-based)

        Returns:
            Numpy array of keypoints for backward compatibility
        """
        if person_idx >= len(self._keypoints):
            return np.zeros((17 if self.source == "yolo" else 33, 3))

        # Return the data array for backward compatibility
        return self._keypoints[person_idx].data

    def get_person(self, person_idx: int = 0) -> Optional[Keypoint]:
        """
        Get Keypoint object for a specific person.

        Args:
            person_idx: Index of the person (0-based)

        Returns:
            Keypoint object for the specified person or None
        """
        if person_idx >= len(self._keypoints):
            return None
        return self._keypoints[person_idx]

    def get_keypoint(self, index: int, person_idx: int = 0) -> np.ndarray:
        """
        Get a specific keypoint by index from a specific person.

        Args:
            index: Keypoint index
            person_idx: Person index (default: 0)

        Returns:
            Array [x, y, confidence] for the keypoint
        """
        if person_idx >= len(self._keypoints):
            return np.array([0.0, 0.0, 0.0])
        return self._keypoints[person_idx].get_keypoint(index)

    def get_keypoint_by_name(self, name: str, person_idx: int = 0) -> np.ndarray:
        """
        Get a keypoint by name (COCO keypoints only) from a specific person.

        Args:
            name: Keypoint name (e.g., "nose", "left_shoulder")
            person_idx: Person index (default: 0)

        Returns:
            Array [x, y, confidence] for the keypoint
        """
        if person_idx >= len(self._keypoints):
            return np.array([0.0, 0.0, 0.0])
        return self._keypoints[person_idx].get_keypoint_by_name(name)

    def filter_by_confidence(self, min_confidence: float) -> "Keypoints":
        """
        Filter keypoints based on person confidence and body part visibility.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            New Keypoints object with filtered Keypoint objects
        """
        if not self._keypoints:
            return self

        # Filter persons based on confidence and body part visibility
        valid_keypoints = []
        for kp in self._keypoints:
            # Check overall confidence
            has_conf = kp.confidence >= min_confidence
            # Check body parts using Keypoint methods
            has_arm = kp.has_visible_arm(min_confidence)
            has_leg = kp.has_visible_leg(min_confidence)

            if has_conf and has_arm and has_leg:
                valid_keypoints.append(kp.data)

        # Create filtered array
        if valid_keypoints:
            if len(valid_keypoints) > 1:
                filtered_keypoints = np.array(valid_keypoints)
            else:
                filtered_keypoints = valid_keypoints[0]
        else:
            # No persons meet criteria, return empty keypoints
            num_kpts = 17 if self.source == "yolo" else 33
            filtered_keypoints = np.zeros((0, num_kpts, 3))

        # Create new Keypoints object with filtered data
        filtered = Keypoints(
            filtered_keypoints,
            source=self.source,
            image_shape=self.image_shape,
            normalize=False,
        )
        return filtered

    def print_keypoints(self) -> None:
        """
        Print keypoints in a readable format for all persons.
        """
        if not self._keypoints:
            print("No keypoints available")
            return

        print(f"Keypoints from {self.source.upper()} detector:")
        print(f"Number of persons detected: {len(self._keypoints)}")
        print("-" * 50)

        for person_idx, kp in enumerate(self._keypoints):
            print(f"\nPerson {person_idx}:")
            kp.print_keypoints()

        print("-" * 50)

    def save(self, csv_writer, label=0) -> None:
        """
        Save keypoints data directly to CSV file for all persons.

        Args:
            csv_writer: CSV writer object
            label: Label for the data
        """
        if not self._keypoints:
            return

        try:
            # Save keypoints for each person
            for kp in self._keypoints:
                row = [label]
                for point in kp.data:
                    row.append(point[0])  # x coordinate
                    row.append(point[1])  # y coordinate
                csv_writer.writerow(row)

        except Exception as e:
            print(f"Error saving keypoints to CSV: {e}")

    def get_kpts_xy(self):
        """
        Get x,y coordinates for all persons.

        Returns:
            List of flattened x,y coordinates for each person
        """
        if not self._keypoints:
            return []
        return [kp.get_kpts_xy() for kp in self._keypoints]


class PoseDetector(ABC):
    """Abstract base class for pose detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray, min_confidence: float = 0.0) -> Keypoints:
        """
        Detect pose in the given frame.

        Args:
            frame: Input frame (BGR format)
            min_confidence: Minimum confidence threshold for filtering keypoints

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
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],  # legs and center
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],  # body and arms
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],  # arms, face
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],  # face to shoulders to arms
    ]

    # Colors for different body parts
    pose_palette = np.array(
        [
            [51, 153, 255],
            [51, 153, 255],
            [51, 153, 255],
            [51, 153, 255],
            [51, 153, 255],  # legs
            [255, 51, 51],
            [255, 51, 51],
            [255, 51, 51],
            [255, 102, 66],
            [255, 102, 66],  # body and arms
            [255, 102, 66],
            [255, 102, 66],
            [51, 153, 51],
            [51, 153, 51],
            [51, 153, 51],  # arms and face
            [51, 153, 51],
            [51, 153, 51],
            [51, 153, 51],
            [51, 153, 51],  # face to shoulders
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

    def detect(self, frame: np.ndarray, min_confidence: float = 0.0) -> Keypoints:
        """
        Detect pose using YOLO model.

        Args:
            frame: Input frame (BGR format)
            min_confidence: Minimum confidence threshold for filtering keypoints

        Returns:
            Keypoints object with normalized and filtered pose data for all detected persons
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
            return Keypoints(
                np.zeros((0, 17, 3)), source="yolo", image_shape=frame.shape[:2]
            )

        keypoints = Keypoints(keypoints_data, source="yolo", image_shape=frame.shape[:2])
        
        # Apply confidence filtering if threshold is set
        if min_confidence > 0.0:
            keypoints = keypoints.filter_by_confidence(min_confidence)
        
        return keypoints

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

        # Draw skeleton and bbox for each person using Keypoint methods
        for kp in keypoints:
            annotated_frame = kp.draw_skeleton(annotated_frame, self.skeleton, self.pose_palette)
            annotated_frame = kp.draw_bbox(annotated_frame)

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

    def detect(self, frame: np.ndarray, min_confidence: float = 0.0) -> Keypoints:
        """
        Detect pose using MediaPipe.

        Args:
            frame: Input frame (BGR format)
            min_confidence: Minimum confidence threshold for filtering keypoints

        Returns:
            Keypoints object with normalized and filtered pose data
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            keypoints = Keypoints(results.pose_landmarks, source="mediapipe")
        else:
            # Return empty keypoints if no detection
            keypoints = Keypoints(None, source="mediapipe")
        
        # Apply confidence filtering if threshold is set
        if min_confidence > 0.0:
            keypoints = keypoints.filter_by_confidence(min_confidence)
        
        return keypoints

    def visualize(
        self, frame: np.ndarray, keypoints: Keypoints, fall_detected: bool = False
    ) -> np.ndarray:
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

        # Draw landmarks for each person (MediaPipe only detects one person)
        for kp in keypoints:
            annotated_frame = kp.draw_mediapipe_landmarks(
                annotated_frame,
                self.mp_drawing,
                self.mp_pose,
                keypoints.original_landmarks,
                fall_detected
            )

        return annotated_frame
