"""
Fall detection module using pose keypoints to detect fallen persons.

Enhanced with camera-aware multi-criteria detection including:
- Body orientation analysis
- Aspect ratio checking
- Keypoint distribution analysis
- Adaptive thresholds based on camera parameters
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict

import numpy as np

PERSPECTIVE_AVAILABLE = False


class FallDetector(ABC):
    """Abstract base class for fall detectors."""

    @abstractmethod
    def detect_fall(self, keypoints) -> Tuple[bool, float, Optional[Dict]]:
        """
        Detect if a person has fallen based on pose keypoints.

        Args:
            keypoints: Keypoints object with normalized pose data

        Returns:
            Tuple of (is_fallen, confidence, details) where:
            - is_fallen: True if fall detected
            - confidence: Detection confidence (0-1)
            - details: Optional dict with detection criteria scores
        """
        pass


class YOLOFallDetector(FallDetector):
    """
    Fall detector implementation for YOLO pose keypoints.
    
    Supports both simple detection (backward compatible) and advanced
    camera-aware multi-criteria detection when CameraConfig is provided.
    """

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

    def __init__(
        self,
        confidence_threshold: float = 0.45,  # Lowered for better sensitivity
        min_keypoints: int = 6,  # Lowered to allow detection with fewer keypoints
        min_keypoint_confidence: float = 0.25  # Lowered to be more permissive
    ):
        """
        Initialize YOLO fall detector.

        Args:
            confidence_threshold: Minimum confidence for fall detection (0-1)
            min_keypoints: Minimum number of visible keypoints required
            min_keypoint_confidence: Minimum confidence for keypoint to be considered visible
        """
        self.confidence_threshold = confidence_threshold
        self.min_keypoints = min_keypoints
        self.min_keypoint_confidence = min_keypoint_confidence
        self.use_advanced_detection = PERSPECTIVE_AVAILABLE
        
        # Criterion weights for multi-criteria fusion - adjusted for better fallen person detection
        self.weights = {
            "orientation": 0.30,
            "aspect_ratio": 0.30,  # Increased weight for aspect ratio (horizontal positioning)
            "height_check": 0.25,
            "distribution": 0.15,  # Slightly reduced but still important
        }

    def detect_fall(
        self,
        keypoints
    ) -> Tuple[bool, float, Optional[Dict]]:
        """
        Detect fall using YOLO keypoints with multi-criteria analysis for all persons.

        Args:
            keypoints: Keypoints object with normalized pose data

        Returns:
            Tuple of (is_fallen, confidence, details) - returns first fall detected
        """
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            return False, 0.0, None

        # Process each person detected
        num_persons = keypoints.num_persons
        for person_idx in range(num_persons):
            person_kpts = keypoints.get_person_keypoints(person_idx)

            # Check if we have sufficient keypoints for reliable detection
            if not self._has_sufficient_keypoints(person_kpts):
                continue  # Skip this person, insufficient data

            if self.use_advanced_detection:
                is_fallen, confidence, details = self._detect_fall_advanced(person_kpts)
            else:
                is_fallen, confidence, details = self._detect_fall_simple(person_kpts)

            if is_fallen:
                # Add person index to details
                if details is None:
                    details = {}
                details["person_idx"] = person_idx
                return is_fallen, confidence, details

        return False, 0.0, None
    
    def _has_sufficient_keypoints(self, keypoints) -> bool:
        """
        Check if there are sufficient visible keypoints for reliable detection.

        Args:
            keypoints: Single person keypoints (17, 3) - normalized numpy array

        Returns:
            True if sufficient keypoints are visible
        """
        # Count keypoints above confidence threshold
        visible_count = np.sum(keypoints[:, 2] > self.min_keypoint_confidence)

        if visible_count < self.min_keypoints:
            return False

        # Additionally, require at least nose OR eyes visible (head)
        # and at least one ankle OR knee (legs)
        head_visible = (
            keypoints[self.NOSE][2] > self.min_keypoint_confidence or
            keypoints[self.LEFT_EYE][2] > self.min_keypoint_confidence or
            keypoints[self.RIGHT_EYE][2] > self.min_keypoint_confidence
        )

        legs_visible = (
            keypoints[self.LEFT_ANKLE][2] > self.min_keypoint_confidence or
            keypoints[self.RIGHT_ANKLE][2] > self.min_keypoint_confidence or
            keypoints[self.LEFT_KNEE][2] > self.min_keypoint_confidence or
            keypoints[self.RIGHT_KNEE][2] > self.min_keypoint_confidence
        )

        return head_visible and legs_visible

    def _detect_fall_simple(
        self,
        keypoints: np.ndarray
    ) -> Tuple[bool, float, Optional[Dict]]:
        """
        Simple fall detection (legacy/backward compatible).

        Args:
            keypoints: Single person keypoints (17, 3) - normalized

        Returns:
            Tuple of (is_fallen, confidence, details)
        """
        """
        Simple fall detection (legacy/backward compatible).
        
        Args:
            keypoints: Single person keypoints (17, 3)
            
        Returns:
            Tuple of (is_fallen, confidence, details)
        """
        # Get key positions with confidence check
        nose = keypoints[self.NOSE]
        left_hip = keypoints[self.LEFT_HIP]
        right_hip = keypoints[self.RIGHT_HIP]
        left_ankle = keypoints[self.LEFT_ANKLE]
        right_ankle = keypoints[self.RIGHT_ANKLE]

        # Check confidence - More permissive thresholds
        if nose[2] < 0.3 or (left_ankle[2] < 0.3 and right_ankle[2] < 0.3):
            return False, 0.0, None
        
        # Check if hips have good confidence
        hips_visible = (left_hip[2] > 0.3 or right_hip[2] > 0.3)

        # Calculate vertical positions
        head_y = nose[1]
        feet_y = max(left_ankle[1], right_ankle[1])  # Use higher ankle
        
        # Calculate hip position (average of visible hips)
        if hips_visible:
            if left_hip[2] > 0.3 and right_hip[2] > 0.3:
                hip_y = (left_hip[1] + right_hip[1]) / 2
            elif left_hip[2] > 0.3:
                hip_y = left_hip[1]
            else:
                hip_y = right_hip[1]
        else:
            hip_y = None

        # Check 1: Head below hips (strong indicator of fall)
        # In image coordinates, Y increases downward, so head_y > hip_y means head is lower
        if hip_y is not None and head_y > hip_y:
            # Head is below hips - very likely fallen
            confidence = 0.95  # Even higher confidence
            details = {
                "method": "simple",
                "trigger": "head_below_hips",
                "head_y": float(head_y),
                "hip_y": float(hip_y),
            }
            return True, confidence, details

        # Check 2: Head close to feet vertically (original logic) - More sensitive
        vertical_diff = abs(head_y - feet_y)
        height_threshold = 80  # pixels, increased to be more sensitive to fallen positions

        if vertical_diff < height_threshold:
            confidence = 1.0 - (vertical_diff / height_threshold)
            details = {
                "method": "simple",
                "trigger": "head_close_to_feet",
                "vertical_diff": float(vertical_diff),
                "threshold": height_threshold,
            }
            return True, min(confidence, 1.0), details

        # Additional check: Body alignment that suggests horizontal position
        if hips_visible and nose[2] > 0.3 and left_ankle[2] > 0.3 and right_ankle[2] > 0.3:
            # Calculate the vertical spread between head and feet vs hips
            head_feet_vertical_span = abs(head_y - feet_y)
            # When vertical span is small, person is more likely lying down
            if head_feet_vertical_span < height_threshold * 0.7:  # More permissive
                confidence = 0.7  # Medium-high confidence
                details = {
                    "method": "simple",
                    "trigger": "low_vertical_span",
                    "vertical_span": float(head_feet_vertical_span),
                    "threshold": height_threshold * 0.7,
                }
                return True, confidence, details

        return False, 0.0, None

    def _detect_fall_advanced(
        self,
        keypoints: np.ndarray
    ) -> Tuple[bool, float, Optional[Dict]]:
        """
        Advanced multi-criteria fall detection using perspective calculations.

        Args:
            keypoints: Single person keypoints (17, 3) - normalized

        Returns:
            Tuple of (is_fallen, confidence, details)
        """
        """
        Advanced multi-criteria fall detection using perspective calculations.

        Args:
            keypoints: Single person keypoints (17, 3)

        Returns:
            Tuple of (is_fallen, confidence, details)
        """
        if not PERSPECTIVE_AVAILABLE:
            return self._detect_fall_simple(keypoints)

        # Import perspective functions
        from .perspective import (
            get_adaptive_thresholds,
            calculate_body_orientation,
            calculate_aspect_ratio,
            calculate_keypoint_distribution
        )

        # Get adaptive thresholds based on person distance
        thresholds = get_adaptive_thresholds(keypoints)

        # Calculate body orientation
        orientation = calculate_body_orientation(keypoints)

        # Calculate aspect ratio
        aspect_ratio = calculate_aspect_ratio(keypoints)

        # Calculate keypoint distribution
        distribution = calculate_keypoint_distribution(keypoints)

        # Multi-criteria analysis
        criteria_scores = {}

        # 1. Height check (head-feet distance)
        height_threshold = thresholds["height_threshold"]
        # Get key positions for height check
        nose = keypoints[self.NOSE]
        left_ankle = keypoints[self.LEFT_ANKLE]
        right_ankle = keypoints[self.RIGHT_ANKLE]

        if nose[2] > 0.3 and (left_ankle[2] > 0.3 or right_ankle[2] > 0.3):
            feet_y = max(left_ankle[1], right_ankle[1]) if left_ankle[2] > 0.3 and right_ankle[2] > 0.3 else (
                left_ankle[1] if left_ankle[2] > 0.3 else right_ankle[1]
            )
            head_feet_distance = abs(nose[1] - feet_y)
            height_score = 1.0 if head_feet_distance < height_threshold else 0.0
            criteria_scores["height_check"] = height_score
        else:
            criteria_scores["height_check"] = 0.0

        # 2. Orientation check (body angle from vertical)
        orientation_threshold = thresholds["orientation_threshold"]
        orientation_score = 1.0 if orientation > orientation_threshold else 0.0
        criteria_scores["orientation"] = orientation_score

        # 3. Aspect ratio check (width/height)
        aspect_threshold = thresholds["aspect_ratio_threshold"]
        aspect_score = 1.0 if aspect_ratio > aspect_threshold else 0.0
        criteria_scores["aspect_ratio"] = aspect_score

        # 4. Distribution check (horizontal spread)
        # Fallen people tend to have more horizontal spread
        spread_ratio_threshold = 1.5  # Higher ratio indicates more horizontal distribution
        distribution_score = 1.0 if distribution["spread_ratio"] > spread_ratio_threshold else 0.0
        criteria_scores["distribution"] = distribution_score

        # Weighted fusion of criteria
        fused_confidence = sum(
            criteria_scores[criterion] * self.weights[criterion]
            for criterion in criteria_scores
        )

        # Determine if fall detected
        is_fallen = fused_confidence >= self.confidence_threshold

        details = {
            "method": "advanced",
            "fused_confidence": fused_confidence,
            "criteria_scores": criteria_scores,
            "orientation_angle": orientation,
            "aspect_ratio": aspect_ratio,
            "distribution": distribution,
            "thresholds": thresholds,
        }

        return is_fallen, fused_confidence, details


class MediaPipeFallDetector(FallDetector):
    """
    Fall detector implementation for MediaPipe pose landmarks.
    
    Supports both simple detection (backward compatible) and advanced
    camera-aware multi-criteria detection when CameraConfig is provided.
    """

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

    def __init__(
        self,
        confidence_threshold: float = 0.65
    ):
        """
        Initialize MediaPipe fall detector.

        Args:
            confidence_threshold: Minimum confidence for fall detection (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.use_advanced_detection = PERSPECTIVE_AVAILABLE
        
        # Criterion weights for multi-criteria fusion
        self.weights = {
            "orientation": 0.35,
            "aspect_ratio": 0.30,
            "height_check": 0.35,
        }

    def detect_fall(self, keypoints) -> Tuple[bool, float, Optional[Dict]]:
        """
        Detect fall using MediaPipe landmarks.

        Args:
            keypoints: Keypoints object with normalized pose data

        Returns:
            Tuple of (is_fallen, confidence, details)
        """
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            return False, 0.0, None

        if self.use_advanced_detection:
            return self._detect_fall_advanced(keypoints)
        else:
            return self._detect_fall_simple(keypoints)

    def _detect_fall_simple(self, keypoints) -> Tuple[bool, float, Optional[Dict]]:
        """
        Simple fall detection (legacy/backward compatible).

        Args:
            keypoints: Keypoints object with normalized pose data

        Returns:
            Tuple of (is_fallen, confidence, details)
        """
        # Get keypoints data (already normalized)
        keypoints_data = keypoints.data

        # Extract key points using MediaPipe indices
        nose = keypoints_data[self.NOSE]
        left_shoulder = keypoints_data[self.LEFT_SHOULDER]
        right_shoulder = keypoints_data[self.RIGHT_SHOULDER]
        left_hip = keypoints_data[self.LEFT_HIP]
        right_hip = keypoints_data[self.RIGHT_HIP]
        left_ankle = keypoints_data[self.LEFT_ANKLE]
        right_ankle = keypoints_data[self.RIGHT_ANKLE]

        # Calculate average positions
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2

        # Check 1: Head below hips (strong fall indicator)
        # In normalized coordinates, Y increases downward
        if nose[1] > hip_y:
            confidence = 0.9
            details = {
                "method": "simple",
                "trigger": "head_below_hips",
                "nose_y": float(nose[1]),
                "hip_y": float(hip_y),
            }
            return True, confidence, details

        # Check 2: Body nearly horizontal and head close to feet
        body_tilt = abs(shoulder_y - hip_y)
        head_to_feet = abs(nose[1] - ankle_y)

        # Fall criteria: body nearly horizontal and head close to feet
        tilt_threshold = 0.1  # normalized coordinates
        height_threshold = 0.3

        if body_tilt < tilt_threshold and head_to_feet < height_threshold:
            confidence = 1.0 - max(body_tilt / tilt_threshold, head_to_feet / height_threshold)
            details = {
                "method": "simple",
                "trigger": "horizontal_body",
                "body_tilt": float(body_tilt),
                "head_to_feet": float(head_to_feet),
            }
            return True, min(confidence, 1.0), details

        return False, 0.0, None

    def _detect_fall_advanced(self, keypoints) -> Tuple[bool, float, Optional[Dict]]:
        """
        Advanced multi-criteria fall detection using perspective calculations.

        Args:
            keypoints: Keypoints object with normalized pose data

        Returns:
            Tuple of (is_fallen, confidence, details)
        """
        if not PERSPECTIVE_AVAILABLE:
            return self._detect_fall_simple(keypoints)

        # Get keypoints data (already normalized)
        keypoints_data = keypoints.data

        # Import perspective functions
        from .perspective import (
            get_adaptive_thresholds,
            calculate_body_orientation,
            calculate_aspect_ratio,
            calculate_keypoint_distribution
        )

        # Get adaptive thresholds based on person distance
        thresholds = get_adaptive_thresholds(keypoints_data)

        # Calculate body orientation
        orientation = calculate_body_orientation(keypoints_data)

        # Calculate aspect ratio
        aspect_ratio = calculate_aspect_ratio(keypoints_data)

        # Calculate keypoint distribution
        distribution = calculate_keypoint_distribution(keypoints_data)

        # Multi-criteria analysis
        criteria_scores = {}

        # 1. Height check (head-shoulder distance in normalized coordinates)
        shoulder_y = (keypoints_data[self.LEFT_SHOULDER][1] + keypoints_data[self.RIGHT_SHOULDER][1]) / 2
        hip_y = (keypoints_data[self.LEFT_HIP][1] + keypoints_data[self.RIGHT_HIP][1]) / 2
        ankle_y = (keypoints_data[self.LEFT_ANKLE][1] + keypoints_data[self.RIGHT_ANKLE][1]) / 2

        # In normalized coordinates, smaller Y differences indicate more horizontal positioning
        body_vertical_span = abs(shoulder_y - ankle_y)
        height_threshold_normalized = 0.3  # Normalized threshold
        height_score = 1.0 if body_vertical_span < height_threshold_normalized else 0.0
        criteria_scores["height_check"] = height_score

        # 2. Orientation check (body angle from vertical)
        orientation_threshold = thresholds["orientation_threshold"]
        orientation_score = 1.0 if orientation > orientation_threshold else 0.0
        criteria_scores["orientation"] = orientation_score

        # 3. Aspect ratio check (width/height)
        aspect_threshold = thresholds["aspect_ratio_threshold"]
        aspect_score = 1.0 if aspect_ratio > aspect_threshold else 0.0
        criteria_scores["aspect_ratio"] = aspect_score

        # 4. Distribution check (horizontal spread)
        spread_ratio_threshold = 1.5
        distribution_score = 1.0 if distribution["spread_ratio"] > spread_ratio_threshold else 0.0
        criteria_scores["distribution"] = distribution_score

        # Weighted fusion of criteria
        fused_confidence = sum(
            criteria_scores[criterion] * self.weights[criterion]
            for criterion in criteria_scores
        )

        # Determine if fall detected
        is_fallen = fused_confidence >= self.confidence_threshold

        details = {
            "method": "advanced",
            "fused_confidence": fused_confidence,
            "criteria_scores": criteria_scores,
            "orientation_angle": orientation,
            "aspect_ratio": aspect_ratio,
            "distribution": distribution,
            "thresholds": thresholds,
        }

        return is_fallen, fused_confidence, details

    def _landmarks_to_numpy(self, landmarks: Any) -> np.ndarray:
        """
        Convert MediaPipe landmarks to numpy array format.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Numpy array (33, 3) with x, y, confidence
        """
        # MediaPipe has 33 landmarks
        keypoints = np.zeros((33, 3))
        
        for i, landmark in enumerate(landmarks.landmark):
            keypoints[i] = [landmark.x, landmark.y, landmark.visibility]
        
        return keypoints