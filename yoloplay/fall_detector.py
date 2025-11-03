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

try:
    from .camera_config import CameraConfig
    from .perspective import (
        get_adaptive_thresholds,
        calculate_body_orientation,
        calculate_aspect_ratio,
        calculate_keypoint_distribution,
    )
    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False
    CameraConfig = None


class FallDetector(ABC):
    """Abstract base class for fall detectors."""

    @abstractmethod
    def detect_fall(self, keypoints: Any) -> Tuple[bool, float, Optional[Dict]]:
        """
        Detect if a person has fallen based on pose keypoints.

        Args:
            keypoints: Pose keypoints array (shape depends on implementation)

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
        camera_config: Optional["CameraConfig"] = None,
        confidence_threshold: float = 0.65,
        min_keypoints: int = 8,
        min_keypoint_confidence: float = 0.3
    ):
        """
        Initialize YOLO fall detector.
        
        Args:
            camera_config: Optional camera configuration for perspective-aware detection
            confidence_threshold: Minimum confidence for fall detection (0-1)
            min_keypoints: Minimum number of visible keypoints required
            min_keypoint_confidence: Minimum confidence for keypoint to be considered visible
        """
        self.camera_config = camera_config
        self.confidence_threshold = confidence_threshold
        self.min_keypoints = min_keypoints
        self.min_keypoint_confidence = min_keypoint_confidence
        self.use_advanced_detection = camera_config is not None and PERSPECTIVE_AVAILABLE
        
        # Criterion weights for multi-criteria fusion
        self.weights = {
            "orientation": 0.30,
            "aspect_ratio": 0.25,
            "height_check": 0.25,
            "distribution": 0.20,
        }

    def detect_fall(
        self,
        keypoints: np.ndarray
    ) -> Tuple[bool, float, Optional[Dict]]:
        """
        Detect fall using YOLO keypoints with multi-criteria analysis.

        Args:
            keypoints: YOLO keypoints array (num_persons, 17, 3) - x, y, conf

        Returns:
            Tuple of (is_fallen, confidence, details)
        """
        if keypoints is None or len(keypoints) == 0:
            return False, 0.0, None

        # Process each person detected (return first fall detected)
        for person_kpts in keypoints:
            # Check if we have sufficient keypoints for reliable detection
            if not self._has_sufficient_keypoints(person_kpts):
                continue  # Skip this person, insufficient data
            
            if self.use_advanced_detection:
                is_fallen, confidence, details = self._detect_fall_advanced(person_kpts)
            else:
                is_fallen, confidence, details = self._detect_fall_simple(person_kpts)
            
            if is_fallen:
                return is_fallen, confidence, details

        return False, 0.0, None
    
    def _has_sufficient_keypoints(self, keypoints) -> bool:
        """
        Check if there are sufficient visible keypoints for reliable detection.
        
        Args:
            keypoints: Single person keypoints (17, 3) - can be tensor or numpy
            
        Returns:
            True if sufficient keypoints are visible
        """
        # Import helper to convert tensor to numpy
        from .perspective import _to_numpy
        
        # Convert to numpy if needed
        keypoints = _to_numpy(keypoints)
        
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

        # Check confidence
        if nose[2] < 0.5 or (left_ankle[2] < 0.5 and right_ankle[2] < 0.5):
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
            confidence = 0.9  # High confidence
            details = {
                "method": "simple",
                "trigger": "head_below_hips",
                "head_y": float(head_y),
                "hip_y": float(hip_y),
            }
            return True, confidence, details

        # Check 2: Head close to feet vertically (original logic)
        vertical_diff = abs(head_y - feet_y)
        height_threshold = 50  # pixels, adjust based on image size

        if vertical_diff < height_threshold:
            confidence = 1.0 - (vertical_diff / height_threshold)
            details = {
                "method": "simple",
                "trigger": "head_close_to_feet",
                "vertical_diff": float(vertical_diff),
                "threshold": height_threshold,
            }
            return True, min(confidence, 1.0), details

        return False, 0.0, None

    def _detect_fall_advanced(
        self,
        keypoints: np.ndarray
    ) -> Tuple[bool, float, Optional[Dict]]:
        """
        Advanced multi-criteria fall detection with camera awareness.
        
        Args:
            keypoints: Single person keypoints (17, 3)
            
        Returns:
            Tuple of (is_fallen, confidence, details)
        """
        # Get adaptive thresholds based on camera config
        thresholds = get_adaptive_thresholds(keypoints, self.camera_config)
        
        # Get key keypoints
        nose = keypoints[self.NOSE]
        left_hip = keypoints[self.LEFT_HIP]
        right_hip = keypoints[self.RIGHT_HIP]
        left_ankle = keypoints[self.LEFT_ANKLE]
        right_ankle = keypoints[self.RIGHT_ANKLE]
        
        # Quick Check: Head below hips (strong fall indicator)
        # In image coordinates, Y increases downward, so head_y > hip_y means head is lower
        hips_visible = (left_hip[2] > 0.3 or right_hip[2] > 0.3)
        head_below_hips_score = 0.0
        
        if nose[2] > 0.3 and hips_visible:
            head_y = nose[1]
            
            # Calculate hip Y position
            if left_hip[2] > 0.3 and right_hip[2] > 0.3:
                hip_y = (left_hip[1] + right_hip[1]) / 2
            elif left_hip[2] > 0.3:
                hip_y = left_hip[1]
            else:
                hip_y = right_hip[1]
            
            # If head is below hips, this is a strong fall signal
            if head_y > hip_y:
                # Calculate how far below (for confidence scoring)
                diff = head_y - hip_y
                # Normalize by expected body segment size
                expected_torso = thresholds.get("expected_height_pixels", 100) * 0.4
                head_below_hips_score = min(diff / expected_torso, 1.0)
        
        # Criterion 1: Body Orientation
        orientation_angle = calculate_body_orientation(keypoints, self.camera_config)
        orientation_score = min(orientation_angle / 90.0, 1.0)  # 0=vertical, 1=horizontal
        
        # Criterion 2: Aspect Ratio
        aspect_ratio = calculate_aspect_ratio(keypoints)
        # Map to score: <0.6 = standing (0), 1.5+ = fallen (1)
        if aspect_ratio < 0.6:
            aspect_score = 0.0
        elif aspect_ratio > 1.5:
            aspect_score = 1.0
        else:
            aspect_score = (aspect_ratio - 0.6) / 0.9  # Linear interpolation
        
        # Criterion 3: Vertical Height Check (head to feet distance)
        if nose[2] > 0.3 and max(left_ankle[2], right_ankle[2]) > 0.3:
            head_y = nose[1]
            feet_y = max(left_ankle[1], right_ankle[1])
            vertical_diff = abs(head_y - feet_y)
            
            height_threshold = thresholds["height_threshold"]
            if vertical_diff < height_threshold:
                height_score = 1.0 - (vertical_diff / height_threshold)
            else:
                height_score = 0.0
        else:
            height_score = 0.0
        
        # Criterion 4: Keypoint Distribution
        distribution = calculate_keypoint_distribution(keypoints)
        spread_ratio = distribution["spread_ratio"]
        # High spread_ratio (>1.5) indicates horizontal spread (fallen)
        if spread_ratio > 1.5:
            distribution_score = min((spread_ratio - 1.0) / 1.5, 1.0)
        else:
            distribution_score = 0.0
        
        # If head is significantly below hips, give it strong weight
        # This overrides other criteria to some extent
        if head_below_hips_score > 0.5:
            # Boost overall confidence when head is below hips
            fused_confidence = max(
                head_below_hips_score * 0.85,  # Head below hips is 85% confidence alone
                (
                    self.weights["orientation"] * orientation_score +
                    self.weights["aspect_ratio"] * aspect_score +
                    self.weights["height_check"] * height_score +
                    self.weights["distribution"] * distribution_score
                )
            )
        else:
            # Normal weighted fusion of criteria
            fused_confidence = (
                self.weights["orientation"] * orientation_score +
                self.weights["aspect_ratio"] * aspect_score +
                self.weights["height_check"] * height_score +
                self.weights["distribution"] * distribution_score
            )
        
        # Decision
        is_fallen = fused_confidence >= self.confidence_threshold
        
        # Detailed results
        details = {
            "method": "advanced",
            "fused_confidence": float(fused_confidence),
            "head_below_hips_score": float(head_below_hips_score),
            "orientation_angle": float(orientation_angle),
            "orientation_score": float(orientation_score),
            "aspect_ratio": float(aspect_ratio),
            "aspect_score": float(aspect_score),
            "height_score": float(height_score),
            "distribution_score": float(distribution_score),
            "person_distance": float(thresholds["person_distance"]),
            "adaptive_threshold": float(thresholds["height_threshold"]),
            "weights": self.weights,
        }
        
        return is_fallen, float(fused_confidence), details


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
        camera_config: Optional["CameraConfig"] = None,
        confidence_threshold: float = 0.65
    ):
        """
        Initialize MediaPipe fall detector.
        
        Args:
            camera_config: Optional camera configuration for perspective-aware detection
            confidence_threshold: Minimum confidence for fall detection (0-1)
        """
        self.camera_config = camera_config
        self.confidence_threshold = confidence_threshold
        self.use_advanced_detection = camera_config is not None and PERSPECTIVE_AVAILABLE
        
        # Criterion weights for multi-criteria fusion
        self.weights = {
            "orientation": 0.35,
            "aspect_ratio": 0.30,
            "height_check": 0.35,
        }

    def detect_fall(self, landmarks: Any) -> Tuple[bool, float, Optional[Dict]]:
        """
        Detect fall using MediaPipe landmarks.

        Args:
            landmarks: MediaPipe pose landmarks object

        Returns:
            Tuple of (is_fallen, confidence, details)
        """
        if not landmarks:
            return False, 0.0, None

        if self.use_advanced_detection:
            return self._detect_fall_advanced(landmarks)
        else:
            return self._detect_fall_simple(landmarks)

    def _detect_fall_simple(self, landmarks: Any) -> Tuple[bool, float, Optional[Dict]]:
        """
        Simple fall detection (legacy/backward compatible).
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Tuple of (is_fallen, confidence, details)
        """
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

        # Check 1: Head below hips (strong fall indicator)
        # In normalized coordinates, Y increases downward
        if nose.y > hip_y:
            confidence = 0.9
            details = {
                "method": "simple",
                "trigger": "head_below_hips",
                "nose_y": float(nose.y),
                "hip_y": float(hip_y),
            }
            return True, confidence, details

        # Check 2: Body nearly horizontal and head close to feet
        body_tilt = abs(shoulder_y - hip_y)
        head_to_feet = abs(nose.y - ankle_y)

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

    def _detect_fall_advanced(self, landmarks: Any) -> Tuple[bool, float, Optional[Dict]]:
        """
        Advanced multi-criteria fall detection with camera awareness.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Tuple of (is_fallen, confidence, details)
        """
        # Convert MediaPipe landmarks to numpy array for perspective calculations
        keypoints_np = self._landmarks_to_numpy(landmarks)
        
        # Get adaptive thresholds
        thresholds = get_adaptive_thresholds(keypoints_np, self.camera_config)
        
        # Get key landmarks
        nose = landmarks.landmark[self.NOSE]
        left_shoulder = landmarks.landmark[self.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.LEFT_HIP]
        right_hip = landmarks.landmark[self.RIGHT_HIP]
        left_ankle = landmarks.landmark[self.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.RIGHT_ANKLE]
        
        # Calculate key positions
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2
        ankle_y = (left_ankle.y + right_ankle.y) / 2
        
        # Check: Head below hips (strong fall indicator)
        head_below_hips_score = 0.0
        if nose.y > hip_mid_y:
            # Head is below hips - calculate score based on how far
            diff = nose.y - hip_mid_y
            expected_torso = 0.3  # Expected torso height in normalized coords
            head_below_hips_score = min(diff / expected_torso, 1.0)
        
        # Criterion 1: Body Orientation (using MediaPipe normalized coords)
        body_vertical_span = abs(shoulder_mid_y - hip_mid_y)
        
        # Map to orientation score (small span = horizontal body)
        orientation_score = 1.0 - min(body_vertical_span / 0.3, 1.0)
        
        # Criterion 2: Aspect Ratio
        aspect_ratio = calculate_aspect_ratio(keypoints_np)
        if aspect_ratio < 0.6:
            aspect_score = 0.0
        elif aspect_ratio > 1.5:
            aspect_score = 1.0
        else:
            aspect_score = (aspect_ratio - 0.6) / 0.9
        
        # Criterion 3: Height Check (head to feet distance)
        head_to_feet = abs(nose.y - ankle_y)
        
        # Normalized coordinate threshold (adjusted for camera)
        height_threshold_norm = 0.4
        if head_to_feet < height_threshold_norm:
            height_score = 1.0 - (head_to_feet / height_threshold_norm)
        else:
            height_score = 0.0
        
        # If head is significantly below hips, boost confidence
        if head_below_hips_score > 0.5:
            fused_confidence = max(
                head_below_hips_score * 0.85,  # Head below hips is strong signal
                (
                    self.weights["orientation"] * orientation_score +
                    self.weights["aspect_ratio"] * aspect_score +
                    self.weights["height_check"] * height_score
                )
            )
        else:
            # Normal weighted fusion
            fused_confidence = (
                self.weights["orientation"] * orientation_score +
                self.weights["aspect_ratio"] * aspect_score +
                self.weights["height_check"] * height_score
            )
        
        # Decision
        is_fallen = fused_confidence >= self.confidence_threshold
        
        # Detailed results
        details = {
            "method": "advanced",
            "fused_confidence": float(fused_confidence),
            "head_below_hips_score": float(head_below_hips_score),
            "orientation_score": float(orientation_score),
            "aspect_ratio": float(aspect_ratio),
            "aspect_score": float(aspect_score),
            "height_score": float(height_score),
            "person_distance": float(thresholds["person_distance"]),
            "weights": self.weights,
        }
        
        return is_fallen, float(fused_confidence), details

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