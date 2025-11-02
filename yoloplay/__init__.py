"""
YoloPlay - Pose detection with YOLO and MediaPipe support.
"""

from .detectors import PoseDetector, YOLOPoseDetector, MediaPipePoseDetector
from .frame_providers import (
    FrameProvider,
    CameraFrameProvider,
    VideoFrameProvider,
    ImageFrameProvider,
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
    "PlaybackMode",
    # Main Processor
    "PoseProcessor",
]

__version__ = "0.1.0"