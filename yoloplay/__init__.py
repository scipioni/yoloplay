"""
YoloPlay - Pose detection with YOLO and MediaPipe support.

Enhanced with camera-aware fall detection using multi-criteria analysis.
"""

from .detectors import PoseDetector, YOLOPoseDetector, MediaPipePoseDetector
from .frame_providers import (
    FrameProvider,
    CameraFrameProvider,
    VideoFrameProvider,
    ImageFrameProvider,
    RTSPFrameProvider,
    PlaybackMode,
)
from .main import PoseProcessor


__all__ = [
    # Detectors
    "PoseDetector",
    "YOLOPoseDetector",
    "MediaPipePoseDetector",
    # Frame Providers
    "FrameProvider",
    "CameraFrameProvider",
    "VideoFrameProvider",
    "ImageFrameProvider",
    "RTSPFrameProvider",
    "PlaybackMode",
    # Main Processor
    "PoseProcessor",
]

__version__ = "0.2.0"