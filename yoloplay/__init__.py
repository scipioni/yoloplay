"""
YoloPlay - Pose detection with YOLO and MediaPipe support.

Enhanced with camera-aware fall detection using multi-criteria analysis.
"""

from .detectors import PoseDetector, YOLOPoseDetector, MediaPipePoseDetector
from .fall_detector import FallDetector, YOLOFallDetector, MediaPipeFallDetector
from .frame_providers import (
    FrameProvider,
    CameraFrameProvider,
    VideoFrameProvider,
    ImageFrameProvider,
    RTSPFrameProvider,
    PlaybackMode,
)
from .main import PoseProcessor

# Optional camera configuration imports
try:
    from .camera_config import CameraConfig, CameraConfigManager, load_camera_config
    _camera_config_exports = ["CameraConfig", "CameraConfigManager", "load_camera_config"]
except ImportError:
    _camera_config_exports = []

__all__ = [
    # Detectors
    "PoseDetector",
    "YOLOPoseDetector",
    "MediaPipePoseDetector",
    # Fall Detectors
    "FallDetector",
    "YOLOFallDetector",
    "MediaPipeFallDetector",
    # Frame Providers
    "FrameProvider",
    "CameraFrameProvider",
    "VideoFrameProvider",
    "ImageFrameProvider",
    "RTSPFrameProvider",
    "PlaybackMode",
    # Main Processor
    "PoseProcessor",
] + _camera_config_exports

__version__ = "0.2.0"