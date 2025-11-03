"""
Fall detection module using pose keypoints to detect fallen persons.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np


class FallDetector(ABC):
    """Abstract base class for fall detectors."""

    @abstractmethod
    def detect_fall(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if a person has fallen based on pose keypoints.

        Args:
            keypoints: Pose keypoints array (shape depends on implementation)

        Returns:
            Tuple of (is_fallen, confidence) where is_fallen is True if fall detected
        """
        pass


class YOLOFallDetector(FallDetector):
    """Fall detector implementation for YOLO pose keypoints."""

    # COCO keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def detect_fall(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Detect fall using YOLO keypoints.

        Logic: Check if the person is lying down by comparing vertical positions
        of head (nose) and feet (ankles). If the difference is small, likely fallen.

        Args:
            keypoints: YOLO keypoints array (num_persons, 17, 3) - x, y, conf

        Returns:
            Tuple of (is_fallen, confidence)
        """
        if keypoints is None or len(keypoints) == 0:
            return False, 0.0

        # Process each person detected
        for person_kpts in keypoints:
            # Get key positions with confidence check
            nose = person_kpts[self.NOSE]
            left_ankle = person_kpts[self.LEFT_ANKLE]
            right_ankle = person_kpts[self.RIGHT_ANKLE]

            # Check confidence
            if nose[2] < 0.5 or left_ankle[2] < 0.5 or right_ankle[2] < 0.5:
                continue

            # Calculate vertical positions
            head_y = nose[1]
            feet_y = max(left_ankle[1], right_ankle[1])  # Use higher ankle

            # If head is below feet or very close vertically, likely fallen
            vertical_diff = abs(head_y - feet_y)
            height_threshold = 50  # pixels, adjust based on image size

            if vertical_diff < height_threshold:
                confidence = 1.0 - (vertical_diff / height_threshold)
                return True, min(confidence, 1.0)

        return False, 0.0


class MediaPipeFallDetector(FallDetector):
    """Fall detector implementation for MediaPipe pose landmarks."""

    # MediaPipe pose landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def detect_fall(self, landmarks) -> Tuple[bool, float]:
        """
        Detect fall using MediaPipe landmarks.

        Logic: Check body orientation by comparing shoulder and hip positions,
        and vertical distance between head and feet.

        Args:
            landmarks: MediaPipe pose landmarks object

        Returns:
            Tuple of (is_fallen, confidence)
        """
        if not landmarks:
            return False, 0.0

        # Extract key points
        nose = landmarks.landmark[self.NOSE]
        left_shoulder = landmarks.landmark[self.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.LEFT_HIP]
        right_hip = landmarks.landmark[self.RIGHT_HIP]
        left_ankle = landmarks.landmark[self.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.RIGHT_ANKLE]

        # Calculate average positions
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        ankle_y = (left_ankle.y + right_ankle.y) / 2

        # Check if body is horizontal (shoulders and hips at similar height)
        body_tilt = abs(shoulder_y - hip_y)

        # Check vertical distance from head to feet
        head_to_feet = abs(nose.y - ankle_y)

        # Fall criteria: body nearly horizontal and head close to feet
        tilt_threshold = 0.1  # normalized coordinates
        height_threshold = 0.3

        if body_tilt < tilt_threshold and head_to_feet < height_threshold:
            confidence = 1.0 - max(body_tilt / tilt_threshold, head_to_feet / height_threshold)
            return True, min(confidence, 1.0)

        return False, 0.0