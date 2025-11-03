"""
Perspective correction module for camera-aware fall detection.

This module provides geometric transformations and calculations to:
- Estimate person distance from camera
- Convert pixel coordinates to real-world distances
- Calculate adaptive thresholds based on perspective
- Compensate for camera height and tilt angle
"""

import math
from typing import Optional, Tuple

import numpy as np

from .camera_config import CameraConfig


def _to_numpy(tensor_or_array):
    """
    Convert PyTorch tensor or array to numpy array.
    
    Args:
        tensor_or_array: Input data (numpy array or PyTorch tensor)
        
    Returns:
        Numpy array on CPU
    """
    if hasattr(tensor_or_array, 'cpu'):
        # It's a PyTorch tensor, move to CPU and convert
        return tensor_or_array.cpu().numpy()
    elif isinstance(tensor_or_array, np.ndarray):
        # Already a numpy array
        return tensor_or_array
    else:
        # Try to convert to numpy
        return np.array(tensor_or_array)


def estimate_person_distance(
    keypoints,
    camera_config: CameraConfig,
    average_person_height: float = 1.7
) -> float:
    """
    Estimate distance from camera to person using pose keypoints.
    
    Uses the person's apparent height in the image compared to expected
    height to estimate distance, accounting for camera parameters.
    
    Args:
        keypoints: Pose keypoints array (17, 3) for YOLO - x, y, confidence
                   Can be numpy array or PyTorch tensor
        camera_config: Camera configuration with FOV and mounting parameters
        average_person_height: Expected real-world height in meters
        
    Returns:
        Estimated distance in meters
    """
    # Convert to numpy if needed
    keypoints = _to_numpy(keypoints)
    
    # Get relevant keypoints for height calculation
    # For YOLO pose: 0=nose, 15=left_ankle, 16=right_ankle
    nose = keypoints[0]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    
    # Check confidence
    if nose[2] < 0.3 or (left_ankle[2] < 0.3 and right_ankle[2] < 0.3):
        # Fallback: use bounding box height estimation
        return _estimate_from_bbox(keypoints, camera_config, average_person_height)
    
    # Calculate person's apparent height in pixels
    head_y = nose[1]
    
    # Use the visible ankle (highest confidence)
    if left_ankle[2] > right_ankle[2]:
        feet_y = left_ankle[1]
    else:
        feet_y = right_ankle[1]
    
    apparent_height_pixels = abs(feet_y - head_y)
    
    if apparent_height_pixels < 10:  # Too small, unreliable
        return _estimate_from_bbox(keypoints, camera_config, average_person_height)
    
    # Calculate distance using perspective projection formula
    # distance = (real_height * focal_length) / pixel_height
    _, fy = camera_config.get_focal_length_pixels()
    
    # Account for camera tilt - adjust expected height
    tilt_rad = math.radians(camera_config.tilt_angle_degrees)
    height_adjustment = math.cos(tilt_rad)
    
    distance = (average_person_height * fy * height_adjustment) / apparent_height_pixels
    
    # Clamp to reasonable range
    return np.clip(distance, 0.5, 20.0)


def _estimate_from_bbox(
    keypoints,
    camera_config: CameraConfig,
    average_person_height: float
) -> float:
    """
    Fallback distance estimation using bounding box.
    
    Args:
        keypoints: Pose keypoints array (will be converted to numpy)
        camera_config: Camera configuration
        average_person_height: Expected person height in meters
        
    Returns:
        Estimated distance in meters
    """
    # Convert to numpy if needed
    keypoints = _to_numpy(keypoints)
    
    # Get bounding box from keypoints
    valid_points = keypoints[keypoints[:, 2] > 0.3]
    
    if len(valid_points) == 0:
        return 3.0  # Default distance
    
    y_coords = valid_points[:, 1]
    bbox_height = np.max(y_coords) - np.min(y_coords)
    
    if bbox_height < 10:
        return 3.0  # Default distance
    
    _, fy = camera_config.get_focal_length_pixels()
    tilt_rad = math.radians(camera_config.tilt_angle_degrees)
    height_adjustment = math.cos(tilt_rad)
    
    distance = (average_person_height * fy * height_adjustment) / bbox_height
    return np.clip(distance, 0.5, 20.0)


def pixel_to_world_distance(
    pixel_distance: float,
    person_distance: float,
    camera_config: CameraConfig,
    axis: str = "vertical"
) -> float:
    """
    Convert pixel distance to real-world distance at given depth.
    
    Args:
        pixel_distance: Distance in pixels
        person_distance: Estimated distance from camera in meters
        camera_config: Camera configuration
        axis: "vertical" or "horizontal" for FOV selection
        
    Returns:
        Real-world distance in meters
    """
    if axis == "vertical":
        fov = camera_config.vertical_fov_degrees
        image_size = camera_config.image_height
    else:
        fov = camera_config.horizontal_fov_degrees
        image_size = camera_config.image_width
    
    # Calculate real-world size of one pixel at given distance
    fov_rad = math.radians(fov)
    view_size_at_distance = 2 * person_distance * math.tan(fov_rad / 2)
    meters_per_pixel = view_size_at_distance / image_size
    
    return pixel_distance * meters_per_pixel


def calculate_expected_body_height(
    person_distance: float,
    camera_config: CameraConfig,
    real_height: float = 1.7
) -> float:
    """
    Calculate expected body height in pixels at given distance.
    
    Args:
        person_distance: Distance from camera in meters
        camera_config: Camera configuration
        real_height: Real-world height in meters
        
    Returns:
        Expected height in pixels
    """
    _, fy = camera_config.get_focal_length_pixels()
    tilt_rad = math.radians(camera_config.tilt_angle_degrees)
    height_adjustment = math.cos(tilt_rad)
    
    expected_pixels = (real_height * fy * height_adjustment) / person_distance
    return expected_pixels


def get_adaptive_thresholds(
    keypoints,
    camera_config: CameraConfig,
    base_threshold_standing: float = 100.0,
    reference_distance: float = 3.0
) -> dict:
    """
    Calculate adaptive thresholds based on person's position and camera setup.
    
    Args:
        keypoints: Pose keypoints array (will be converted to numpy)
        camera_config: Camera configuration
        base_threshold_standing: Base pixel threshold at reference distance
        reference_distance: Reference distance for base threshold (meters)
        
    Returns:
        Dictionary with adaptive threshold values:
        - height_threshold: Pixels for head-feet distance check
        - aspect_ratio_threshold: Expected ratio for fallen person
        - orientation_threshold: Degrees for body angle check
        - person_distance: Estimated distance in meters
    """
    # Convert to numpy if needed
    keypoints = _to_numpy(keypoints)
    
    # Estimate person distance
    person_distance = estimate_person_distance(keypoints, camera_config)
    
    # Scale threshold based on distance (closer = larger pixels, further = smaller)
    distance_scale = reference_distance / person_distance
    height_threshold = base_threshold_standing * distance_scale
    
    # Account for camera tilt
    tilt_factor = math.cos(math.radians(camera_config.tilt_angle_degrees))
    height_threshold *= tilt_factor
    
    # Calculate expected body dimensions
    expected_height_px = calculate_expected_body_height(
        person_distance, camera_config, real_height=1.7
    )
    
    # Aspect ratio threshold (width/height for fallen person)
    # Standing: ~0.3-0.5, Fallen: ~1.5-3.0
    aspect_ratio_threshold = 1.2  # Threshold between standing and fallen
    
    # Orientation threshold (degrees from vertical)
    # Standing: 0-20°, Fallen: 60-90°
    orientation_threshold = 45.0  # Midpoint threshold
    
    return {
        "height_threshold": float(height_threshold),
        "aspect_ratio_threshold": aspect_ratio_threshold,
        "orientation_threshold": orientation_threshold,
        "person_distance": float(person_distance),
        "expected_height_pixels": float(expected_height_px),
    }


def calculate_body_orientation(
    keypoints,
    camera_config: CameraConfig
) -> float:
    """
    Calculate body orientation angle relative to vertical.
    
    Computes the angle of the torso (shoulders to hips) relative to vertical,
    accounting for camera tilt.
    
    Args:
        keypoints: Pose keypoints array (17, 3) for YOLO (will be converted to numpy)
        camera_config: Camera configuration
        
    Returns:
        Angle in degrees from vertical (0° = standing, 90° = horizontal)
    """
    # Convert to numpy if needed
    keypoints = _to_numpy(keypoints)
    
    # YOLO keypoints: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    # Check confidence
    confidences = [left_shoulder[2], right_shoulder[2], left_hip[2], right_hip[2]]
    if min(confidences) < 0.3:
        return 0.0  # Can't determine, assume standing
    
    # Calculate midpoints
    shoulder_mid = np.array([
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    ])
    
    hip_mid = np.array([
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    ])
    
    # Calculate body vector (from hips to shoulders)
    body_vector = shoulder_mid - hip_mid
    
    # Calculate angle from vertical (vertical in image coordinates is negative Y)
    # In image coords: Y increases downward, so vertical is (0, 1)
    vertical = np.array([0, 1])
    
    # Calculate angle using dot product
    dot_product = np.dot(body_vector, vertical)
    magnitude = np.linalg.norm(body_vector) * np.linalg.norm(vertical)
    
    if magnitude == 0:
        return 0.0
    
    # Angle from vertical
    angle_rad = np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0))
    angle_deg = math.degrees(angle_rad)
    
    # Subtract camera tilt to get true orientation
    # If camera tilts down 45°, a standing person appears at 45° in image
    corrected_angle = abs(angle_deg - camera_config.tilt_angle_degrees)
    
    return float(corrected_angle)


def calculate_aspect_ratio(keypoints) -> float:
    """
    Calculate bounding box aspect ratio of detected person.
    
    Args:
        keypoints: Pose keypoints array (will be converted to numpy)
        
    Returns:
        Width/Height ratio (>1 for fallen, <1 for standing)
    """
    # Convert to numpy if needed
    keypoints = _to_numpy(keypoints)
    
    # Get bounding box from keypoints
    valid_points = keypoints[keypoints[:, 2] > 0.3]
    
    if len(valid_points) < 4:
        return 0.5  # Default to standing-ish ratio
    
    x_coords = valid_points[:, 0]
    y_coords = valid_points[:, 1]
    
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    
    if height == 0:
        return 0.5
    
    return float(width / height)


def calculate_keypoint_distribution(keypoints) -> dict:
    """
    Analyze how keypoints are distributed (horizontal vs vertical).
    
    Args:
        keypoints: Pose keypoints array (will be converted to numpy)
        
    Returns:
        Dictionary with distribution metrics:
        - horizontal_spread: Std dev of x coordinates
        - vertical_spread: Std dev of y coordinates
        - spread_ratio: horizontal/vertical spread
    """
    # Convert to numpy if needed
    keypoints = _to_numpy(keypoints)
    
    valid_points = keypoints[keypoints[:, 2] > 0.3]
    
    if len(valid_points) < 4:
        return {
            "horizontal_spread": 0.0,
            "vertical_spread": 0.0,
            "spread_ratio": 1.0
        }
    
    x_coords = valid_points[:, 0]
    y_coords = valid_points[:, 1]
    
    h_spread = float(np.std(x_coords))
    v_spread = float(np.std(y_coords))
    
    if v_spread == 0:
        ratio = 2.0  # High ratio suggests horizontal spread
    else:
        ratio = h_spread / v_spread
    
    return {
        "horizontal_spread": h_spread,
        "vertical_spread": v_spread,
        "spread_ratio": float(ratio)
    }


def get_ground_plane_position(
    point_y: float,
    camera_config: CameraConfig,
    image_height: int
) -> float:
    """
    Estimate real-world distance on ground plane from pixel Y coordinate.
    
    Useful for determining if a person is near or far from camera based
    on their vertical position in the frame.
    
    Args:
        point_y: Y coordinate in image (pixels from top)
        camera_config: Camera configuration
        image_height: Image height in pixels
        
    Returns:
        Estimated distance along ground plane in meters
    """
    # Normalize Y coordinate (0 = top, 1 = bottom)
    y_normalized = point_y / image_height
    
    # Account for camera tilt and height
    tilt_rad = math.radians(camera_config.tilt_angle_degrees)
    
    # Simple geometric estimation
    # This is a simplified model; more complex projections possible
    vertical_fov_rad = math.radians(camera_config.vertical_fov_degrees)
    
    # Angle to point from camera's optical axis
    angle_from_center = (y_normalized - 0.5) * vertical_fov_rad
    angle_to_ground = tilt_rad + angle_from_center
    
    if angle_to_ground <= 0:
        return 20.0  # Beyond horizon, return max distance
    
    # Distance = height / tan(angle)
    distance = camera_config.height_meters / math.tan(angle_to_ground)
    
    return np.clip(distance, 0.5, 20.0)