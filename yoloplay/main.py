"""
Main application for pose detection with various input sources.
"""

import os
import time
from typing import Optional

import cv2

from .detectors import PoseDetector, YOLOPoseDetector, MediaPipePoseDetector
from .frame_providers import (
    FrameProvider,
    CameraFrameProvider,
    VideoFrameProvider,
    ImageFrameProvider,
    PlaybackMode,
)


class PoseProcessor:
    """
    Main processor class that combines a pose detector with a frame provider.
    """

    def __init__(self, detector: PoseDetector, frame_provider: FrameProvider):
        """
        Initialize the pose processor.

        Args:
            detector: Pose detector instance (YOLO or MediaPipe)
            frame_provider: Frame provider instance (camera, video, or images)
        """
        self.detector = detector
        self.frame_provider = frame_provider
        self.display_available = self._check_display_available()
        
        # FPS tracking
        self.prev_frame_time = 0.0
        self.fps = 0.0

    def run(self, window_name: str = "Pose Detection") -> None:
        """
        Run the main processing loop.

        Args:
            window_name: Name of the display window
        """
        # Open the frame source
        if not self.frame_provider.open():
            raise ValueError("Cannot open frame source")

        print(f"Frame source opened successfully.")
        
        # Display controls based on provider type
        if isinstance(self.frame_provider, VideoFrameProvider):
            print("Controls: 'q'=quit, 'p'=play/pause, SPACE=step, 'm'=toggle mode")
        elif isinstance(self.frame_provider, ImageFrameProvider):
            print("Controls: 'q'=quit, 'n'=next, 'p'=previous, 'm'=toggle mode")
        else:
            print("Controls: 'q'=quit")

        try:
            while True:
                # Read frame from provider
                ret, frame = self.frame_provider.read()

                if not ret:
                    print("End of frames or failed to read frame")
                    break

                # If frame is None (paused or waiting for step), handle input and continue
                if frame is None:
                    if self.display_available:
                        key = cv2.waitKey(30) & 0xFF
                        self._handle_key_press(key)
                        if key == ord('q'):
                            break
                    else:
                        time.sleep(0.03)
                    continue

                # Calculate FPS
                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                self.prev_frame_time = current_time
                
                if time_diff > 0:
                    self.fps = 1.0 / time_diff
                
                # Detect pose
                results = self.detector.detect(frame)

                # Visualize results
                annotated_frame = self.detector.visualize(frame, results)

                # Display the frame if display is available
                if self.display_available:
                    # Add FPS counter
                    fps_text = f"FPS: {self.fps:.1f}"
                    cv2.putText(
                        annotated_frame,
                        fps_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    
                    # Add status text for video/image providers
                    status_text = self._get_status_text()
                    if status_text:
                        cv2.putText(
                            annotated_frame,
                            status_text,
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                    cv2.imshow(window_name, annotated_frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    self._handle_key_press(key)
                else:
                    # In headless mode, just add a small delay
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Release resources
            self.frame_provider.release()
            if self.display_available:
                cv2.destroyAllWindows()

    def _handle_key_press(self, key: int) -> None:
        """
        Handle keyboard input for controlling playback.

        Args:
            key: ASCII code of the pressed key
        """
        if isinstance(self.frame_provider, VideoFrameProvider):
            if key == ord('p'):
                self.frame_provider.toggle_pause()
                status = "Paused" if self.frame_provider.is_paused else "Playing"
                print(f"Video {status}")
            elif key == ord(' '):  # Space key
                self.frame_provider.step()
            elif key == ord('m'):
                # Toggle between PLAY and STEP mode
                new_mode = (
                    PlaybackMode.STEP
                    if self.frame_provider.mode == PlaybackMode.PLAY
                    else PlaybackMode.PLAY
                )
                self.frame_provider.set_mode(new_mode)
                print(f"Mode changed to {new_mode.value}")

        elif isinstance(self.frame_provider, ImageFrameProvider):
            if key == ord('n') or key == ord(' '):  # Next image
                self.frame_provider.step()
            elif key == ord('p'):  # Previous image
                self.frame_provider.previous()
            elif key == ord('m'):
                # Toggle between PLAY and STEP mode
                new_mode = (
                    PlaybackMode.STEP
                    if self.frame_provider.mode == PlaybackMode.PLAY
                    else PlaybackMode.PLAY
                )
                self.frame_provider.set_mode(new_mode)
                print(f"Mode changed to {new_mode.value}")

    def _get_status_text(self) -> Optional[str]:
        """
        Get status text to display on the frame.

        Returns:
            Status text string or None
        """
        if isinstance(self.frame_provider, VideoFrameProvider):
            if self.frame_provider.mode == PlaybackMode.STEP:
                return "MODE: STEP (SPACE to advance)"
            elif self.frame_provider.is_paused:
                return "PAUSED (Press 'p' to play)"
            else:
                return "PLAYING"
        elif isinstance(self.frame_provider, ImageFrameProvider):
            total = len(self.frame_provider.image_paths)
            current = self.frame_provider.current_index
            mode = self.frame_provider.mode.value.upper()
            return f"Image {current}/{total} - MODE: {mode}"
        return None

    def _check_display_available(self) -> bool:
        """
        Check if a display is available (for GUI operations).
        
        Returns:
            True if display is available, False otherwise
        """
        # Check if running in a container without display
        if os.environ.get("DISPLAY"):
            return True
        # On Linux, try to access X11 display
        try:
            import subprocess

            result = subprocess.run(
                ["xdpyinfo"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return result.returncode == 0
        except FileNotFoundError:
            # xdpyinfo not available, assume no display
            return False
        except:
            # Other error, safer to assume no display
            return False


def main():
    """Command line entry point for pose detection application."""
    import argparse

    parser = argparse.ArgumentParser(description="Pose detection with YOLO or MediaPipe")
    parser.add_argument(
        "--detector",
        type=str,
        choices=["yolo", "mediapipe"],
        default="yolo",
        help="Pose detector to use (default: yolo)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-pose.pt",
        help="YOLO Pose model path (default: yolov8n-pose.pt)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Camera index to use for camera input",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file to process",
    )
    parser.add_argument(
        "--images",
        nargs='+',
        help="List of image files to process",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["play", "step"],
        default="play",
        help="Playback mode for video/images (default: play)",
    )

    args = parser.parse_args()

    # Create detector based on user choice
    if args.detector == "mediapipe":
        detector = MediaPipePoseDetector()
        print("Using MediaPipe pose detector")
    else:
        detector = YOLOPoseDetector(args.model)
        print(f"Using YOLO pose detector with model: {args.model}")

    # Create frame provider based on input source
    playback_mode = PlaybackMode.PLAY if args.mode == "play" else PlaybackMode.STEP

    if args.video:
        frame_provider = VideoFrameProvider(args.video, mode=playback_mode)
        print(f"Processing video: {args.video}")
    elif args.images:
        frame_provider = ImageFrameProvider(args.images, mode=playback_mode)
        print(f"Processing {len(args.images)} images")
    else:
        # Default to camera
        camera_index = args.camera if args.camera is not None else 0
        frame_provider = CameraFrameProvider(camera_index)
        print(f"Using camera index: {camera_index}")

    # Create processor and run
    processor = PoseProcessor(detector, frame_provider)
    processor.run()


if __name__ == "__main__":
    main()
