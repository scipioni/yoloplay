"""
Frame provider module for different input sources (camera, video, images).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


class PlaybackMode(Enum):
    """Playback modes for video and image sources."""
    STEP = "step"  # Step through frames one by one
    PLAY = "play"  # Continuous playback


class FrameProvider(ABC):
    """Abstract base class for frame providers."""

    def __init__(self):
        """Initialize frame provider."""
        self._is_opened = False

    @abstractmethod
    def open(self) -> bool:
        """
        Open the frame source.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.

        Returns:
            Tuple of (success, frame) where success is True if frame was read
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release the frame source and cleanup."""
        pass

    @property
    def is_opened(self) -> bool:
        """Check if the frame source is opened."""
        return self._is_opened


class CameraFrameProvider(FrameProvider):
    """Frame provider for camera input."""

    def __init__(self, camera_index: int = 0):
        """
        Initialize camera frame provider.

        Args:
            camera_index: Index of the camera to use
        """
        super().__init__()
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        self._is_opened = self.cap.isOpened()
        return self._is_opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if not self._is_opened or self.cap is None:
            return False, None
        return self.cap.read()

    def release(self) -> None:
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self._is_opened = False


class VideoFrameProvider(FrameProvider):
    """Frame provider for video file input with play/pause/step controls."""

    def __init__(self, video_path: str, mode: PlaybackMode = PlaybackMode.PLAY):
        """
        Initialize video frame provider.

        Args:
            video_path: Path to the video file
            mode: Initial playback mode (PLAY or STEP)
        """
        super().__init__()
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.mode = mode
        self._paused = False
        self._step_requested = False

    def open(self) -> bool:
        """Open the video file."""
        self.cap = cv2.VideoCapture(self.video_path)
        self._is_opened = self.cap.isOpened()
        return self._is_opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video.

        In STEP mode: Only reads when step is requested
        In PLAY mode: Reads continuously unless paused
        """
        if not self._is_opened or self.cap is None:
            return False, None

        # In STEP mode, only read if step was requested
        if self.mode == PlaybackMode.STEP:
            if self._step_requested:
                self._step_requested = False
                return self.cap.read()
            return True, None  # Return success but no frame (waiting for step)

        # In PLAY mode, read unless paused
        if self._paused:
            return True, None  # Return success but no frame (paused)

        return self.cap.read()

    def step(self) -> None:
        """Request to step to the next frame (used in STEP mode)."""
        self._step_requested = True

    def toggle_pause(self) -> None:
        """Toggle pause state (used in PLAY mode)."""
        self._paused = not self._paused

    def set_mode(self, mode: PlaybackMode) -> None:
        """
        Set the playback mode.

        Args:
            mode: New playback mode (PLAY or STEP)
        """
        self.mode = mode
        if mode == PlaybackMode.PLAY:
            self._paused = False
        self._step_requested = False

    @property
    def is_paused(self) -> bool:
        """Check if video is paused."""
        return self._paused

    def release(self) -> None:
        """Release the video file."""
        if self.cap is not None:
            self.cap.release()
            self._is_opened = False


class ImageFrameProvider(FrameProvider):
    """Frame provider for image file input with navigation controls."""

    def __init__(self, image_paths: list[str], mode: PlaybackMode = PlaybackMode.STEP):
        """
        Initialize image frame provider.

        Args:
            image_paths: List of paths to image files
            mode: Playback mode (STEP or PLAY for auto-advance)
        """
        super().__init__()
        self.image_paths = image_paths
        self.current_index = 0
        self.mode = mode
        self._step_requested = False
        self._current_frame: Optional[np.ndarray] = None

    def open(self) -> bool:
        """Open the first image."""
        if not self.image_paths:
            return False

        self._is_opened = True
        # Load the first image
        self._current_frame = cv2.imread(self.image_paths[self.current_index])
        return self._current_frame is not None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the current image.

        In STEP mode: Returns same image until step is requested
        In PLAY mode: Auto-advances through images
        """
        if not self._is_opened:
            return False, None

        if self.current_index >= len(self.image_paths):
            return False, None  # End of images

        # In STEP mode, only advance if step was requested
        if self.mode == PlaybackMode.STEP:
            if self._step_requested:
                self._step_requested = False
                if self.current_index < len(self.image_paths):
                    self._current_frame = cv2.imread(self.image_paths[self.current_index])
                    self.current_index += 1
            return True, self._current_frame

        # In PLAY mode, return frame and auto-advance
        frame = self._current_frame
        if self.current_index < len(self.image_paths):
            self._current_frame = cv2.imread(self.image_paths[self.current_index])
            self.current_index += 1
        return True, frame

    def step(self) -> None:
        """Request to step to the next image."""
        self._step_requested = True

    def previous(self) -> None:
        """Go to the previous image."""
        if self.current_index > 1:
            self.current_index -= 2  # Go back 2 (one for current, one for previous)
            if self.current_index < 0:
                self.current_index = 0
            self._step_requested = True

    def set_mode(self, mode: PlaybackMode) -> None:
        """
        Set the playback mode.

        Args:
            mode: New playback mode (PLAY or STEP)
        """
        self.mode = mode
        self._step_requested = False

    def release(self) -> None:
        """Release resources."""
        self._is_opened = False
        self._current_frame = None