"""
Main application for pose detection with various input sources.
"""

import os
import time
import json
from typing import Optional, Dict, Any

import cv2

from .detectors import PoseDetector, YOLOPoseDetector, MediaPipePoseDetector
from .fall_detector import FallDetector, YOLOFallDetector, MediaPipeFallDetector
from .frame_providers import (
    FrameProvider,
    CameraFrameProvider,
    VideoFrameProvider,
    RTSPFrameProvider,
    ImageFrameProvider,
    PlaybackMode,
)

try:
    from .camera_config import CameraConfig, load_camera_config
    CAMERA_CONFIG_AVAILABLE = True
except ImportError:
    CAMERA_CONFIG_AVAILABLE = False
    CameraConfig = None


class PoseProcessor:
    """
    Main processor class that combines a pose detector with a frame provider.
    """

    def __init__(
        self,
        detector: PoseDetector,
        frame_provider: FrameProvider,
        fall_detector: Optional[FallDetector] = None,
        camera_config: Optional["CameraConfig"] = None,
        show_debug_info: bool = False,
    ):
        """
        Initialize the pose processor.

        Args:
            detector: Pose detector instance (YOLO or MediaPipe)
            frame_provider: Frame provider instance (camera, video, or images)
            fall_detector: Optional fall detector instance
            camera_config: Optional camera configuration for display
            show_debug_info: Whether to show detailed debug information
        """
        self.detector = detector
        self.frame_provider = frame_provider
        self.fall_detector = fall_detector
        self.camera_config = camera_config
        self.show_debug_info = show_debug_info
        self.display_available = self._check_display_available()

        # FPS tracking
        self.prev_frame_time = 0.0
        self.fps = 0.0
        
        # Fall detection details
        self.fall_details: Optional[Dict] = None

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

                # Detect falls if fall detector is enabled
                fall_detected = False
                fall_confidence = 0.0
                self.fall_details = None
                keypoints_data = None
                
                if self.fall_detector is not None:
                    if isinstance(self.detector, YOLOPoseDetector):
                        # Extract keypoints from YOLO results
                        for r in results:
                            if hasattr(r, "keypoints") and r.keypoints is not None:
                                keypoints = r.keypoints.data
                                keypoints_data = keypoints
                                fall_detected, fall_confidence, self.fall_details = self.fall_detector.detect_fall(keypoints)
                                break  # Process only first person for now
                    elif isinstance(self.detector, MediaPipePoseDetector):
                        # Use MediaPipe results directly
                        if results and results.pose_landmarks:
                            fall_detected, fall_confidence, self.fall_details = self.fall_detector.detect_fall(results.pose_landmarks)
                
                # Output JSON debug information
                if self.show_debug_info:
                    self._output_json_debug(fall_detected, fall_confidence, keypoints_data)

                # Visualize results
                annotated_frame = self.detector.visualize(frame, results, fall_detected)

                # Add fall detection visualization with enhanced details
                if self.fall_detector is not None:
                    annotated_frame = self._add_fall_visualization(
                        annotated_frame, fall_detected, fall_confidence
                    )

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
                    
                    # Add camera config info if available
                    if self.camera_config:
                        info_text = f"Cam: {self.camera_config.name} ({self.camera_config.height_meters}m, {self.camera_config.tilt_angle_degrees}deg)"
                        cv2.putText(
                            annotated_frame,
                            info_text,
                            (10, annotated_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
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

    def _add_fall_visualization(
        self,
        frame,
        fall_detected: bool,
        fall_confidence: float
    ):
        """
        Add enhanced fall detection visualization to frame.
        
        Args:
            frame: Input frame
            fall_detected: Whether fall was detected
            fall_confidence: Detection confidence
            
        Returns:
            Annotated frame
        """
        if fall_detected:
            # Add red alert text with detection mode indicator
            mode_indicator = ""
            if self.fall_details and self.fall_details.get("method") == "advanced":
                mode_indicator = " [ADV]"
            elif self.fall_details and self.fall_details.get("method") == "simple":
                mode_indicator = " [SIMPLE]"
            
            cv2.putText(
                frame,
                f"FALL DETECTED! ({fall_confidence:.2f}){mode_indicator}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
            # Draw red border around frame
            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], frame.shape[0]),
                (0, 0, 255),
                10
            )
            
            # Add detailed criteria if debug mode and details available
            if self.show_debug_info and self.fall_details:
                y_offset = 130
                details = self.fall_details
                
                if details.get("method") == "advanced":
                    # Show individual criterion scores
                    criteria_text = []
                    
                    if "head_below_hips_score" in details:
                        criteria_text.append(f"Head<Hips: {details.get('head_below_hips_score', 0):.2f}")
                    
                    criteria_text.extend([
                        f"Orientation: {details.get('orientation_score', 0):.2f}",
                        f"Aspect: {details.get('aspect_score', 0):.2f}",
                        f"Height: {details.get('height_score', 0):.2f}",
                    ])
                    
                    if "distribution_score" in details:
                        criteria_text.append(f"Distrib: {details.get('distribution_score', 0):.2f}")
                
                elif details.get("method") == "simple" and details.get("trigger") == "head_below_hips":
                    # Show simple mode head below hips trigger
                    criteria_text = [f"Trigger: Head below hips"]
                    
                    for text in criteria_text:
                        cv2.putText(
                            frame,
                            text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )
                        y_offset += 25
                    
                    # Show distance if available
                    if "person_distance" in details:
                        cv2.putText(
                            frame,
                            f"Distance: {details['person_distance']:.1f}m",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )
        else:
            # Add green status text with detection mode indicator
            mode_indicator = ""
            if self.fall_details and self.fall_details.get("method") == "advanced":
                mode_indicator = " [ADV]"
            elif self.fall_details and self.fall_details.get("method") == "simple":
                mode_indicator = " [SIMPLE]"
            
            cv2.putText(
                frame,
                f"No Fall ({fall_confidence:.2f}){mode_indicator}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            
            # Show criteria in debug mode even when no fall
            if self.show_debug_info and self.fall_details:
                y_offset = 120
                details = self.fall_details
                
                if details.get("method") == "advanced":
                    cv2.putText(
                        frame,
                        f"Conf: {details.get('fused_confidence', 0):.2f}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
        
        return frame

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
    
    def _output_json_debug(
        self,
        fall_detected: bool,
        fall_confidence: float,
        keypoints_data: Any
    ) -> None:
        """
        Output JSON debug information to console.
        
        Args:
            fall_detected: Whether fall was detected
            fall_confidence: Detection confidence
            keypoints_data: Raw keypoints data
        """
        debug_info: Dict[str, Any] = {
            "timestamp": time.time(),
            "fallen_state": fall_detected,
            "confidence": float(fall_confidence),
        }
        
        # Add image name if using ImageFrameProvider
        if isinstance(self.frame_provider, ImageFrameProvider):
            if self.frame_provider.current_index > 0:
                current_idx = self.frame_provider.current_index - 1
                if current_idx < len(self.frame_provider.image_paths):
                    debug_info["image_name"] = os.path.basename(
                        self.frame_provider.image_paths[current_idx]
                    )
                    debug_info["image_path"] = self.frame_provider.image_paths[current_idx]
        
        # Add keypoints if available
        if keypoints_data is not None:
            try:
                # Convert tensor to numpy if needed
                import numpy as np
                if hasattr(keypoints_data, 'cpu'):
                    kpts_np = keypoints_data.cpu().numpy()
                else:
                    kpts_np = keypoints_data
                
                # Extract first person keypoints (17, 3) - x, y, conf
                if len(kpts_np.shape) == 3 and kpts_np.shape[0] > 0:
                    person_kpts = kpts_np[0]
                    # Convert to list ensuring it's JSON serializable
                    person_kpts_list = person_kpts.tolist() if hasattr(person_kpts, 'tolist') else person_kpts
                    debug_info["keypoints"] = {
                        "count": int(person_kpts.shape[0]),
                        "data": person_kpts_list,
                        "format": "COCO (17 keypoints: x, y, confidence)"
                    }
            except Exception as e:
                debug_info["keypoints_error"] = str(e)
        
        # Add fall detection details if available
        if self.fall_details:
            debug_info["detection_details"] = {
                "method": self.fall_details.get("method"),
                "adaptive_threshold": float(self.fall_details.get("adaptive_threshold")) if self.fall_details.get("adaptive_threshold") is not None else None,
                "person_distance": float(self.fall_details.get("person_distance")) if self.fall_details.get("person_distance") is not None else None,
            }
            
            # Add criterion scores for advanced detection
            if self.fall_details.get("method") == "advanced":
                debug_info["detection_details"]["criteria"] = {
                    "fused_confidence": float(self.fall_details.get("fused_confidence")) if self.fall_details.get("fused_confidence") is not None else 0.0,
                    "head_below_hips_score": float(self.fall_details.get("head_below_hips_score")) if self.fall_details.get("head_below_hips_score") is not None else 0.0,
                    "orientation_score": float(self.fall_details.get("orientation_score")) if self.fall_details.get("orientation_score") is not None else 0.0,
                    "orientation_angle": float(self.fall_details.get("orientation_angle")) if self.fall_details.get("orientation_angle") is not None else 0.0,
                    "aspect_ratio": float(self.fall_details.get("aspect_ratio")) if self.fall_details.get("aspect_ratio") is not None else 0.0,
                    "aspect_score": float(self.fall_details.get("aspect_score")) if self.fall_details.get("aspect_score") is not None else 0.0,
                    "height_score": float(self.fall_details.get("height_score")) if self.fall_details.get("height_score") is not None else 0.0,
                    "distribution_score": float(self.fall_details.get("distribution_score")) if self.fall_details.get("distribution_score") is not None else 0.0,
                }
            elif self.fall_details.get("method") == "simple":
                debug_info["detection_details"]["trigger"] = self.fall_details.get("trigger")
        
        # Print JSON to console (convert any remaining tensors to floats/strings)
        def serialize_tensors(obj):
            if hasattr(obj, 'cpu'):
                return obj.cpu().numpy().tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, (list, tuple)):
                return [serialize_tensors(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: serialize_tensors(value) for key, value in obj.items()}
            else:
                return obj
        print(json.dumps(serialize_tensors(debug_info), indent=2))


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
    parser.add_argument(
        "--fall-detection",
        action="store_true",
        default=True,
        help="Enable fall detection using pose keypoints",
    )
    
    # Camera configuration parameters
    parser.add_argument(
        "--camera-config",
        type=str,
        default="data/cameras.yaml",
        help="Path to camera configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="fallwebm",
        help="Camera ID to load from configuration file",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        help="Camera height in meters (for inline config)",
    )
    parser.add_argument(
        "--camera-tilt",
        type=float,
        help="Camera tilt angle in degrees (for inline config)",
    )
    parser.add_argument(
        "--camera-fov-h",
        type=float,
        help="Horizontal FOV in degrees (for inline config)",
    )
    parser.add_argument(
        "--camera-fov-v",
        type=float,
        help="Vertical FOV in degrees (for inline config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed debug information and detection criteria",
    )

    args = parser.parse_args()

    # Load camera configuration if provided
    camera_config = None
    if CAMERA_CONFIG_AVAILABLE:
        try:
            if args.camera_config:
                # Load from file
                camera_config = load_camera_config(
                    filepath=args.camera_config,
                    camera_id=args.camera_id
                )
                if camera_config:
                    print(f"Loaded camera config: {camera_config.name}")
                    print(f"  Height: {camera_config.height_meters}m, Tilt: {camera_config.tilt_angle_degrees}°")
                    print(f"  FOV: {camera_config.horizontal_fov_degrees}°×{camera_config.vertical_fov_degrees}°")
            elif args.camera_height and args.camera_tilt and args.camera_fov_h and args.camera_fov_v:
                # Create from inline parameters
                # Get image resolution from video/camera
                image_width = 1920  # Default, will be updated from actual frame
                image_height = 1080
                
                camera_config = load_camera_config(
                    height_meters=args.camera_height,
                    tilt_angle_degrees=args.camera_tilt,
                    horizontal_fov_degrees=args.camera_fov_h,
                    vertical_fov_degrees=args.camera_fov_v,
                    image_width=image_width,
                    image_height=image_height,
                    camera_id="inline_config",
                )
                print(f"Created inline camera config:")
                print(f"  Height: {camera_config.height_meters}m, Tilt: {camera_config.tilt_angle_degrees}°")
                print(f"  FOV: {camera_config.horizontal_fov_degrees}°×{camera_config.vertical_fov_degrees}°")
        except Exception as e:
            print(f"Warning: Failed to load camera config: {e}")
            print("Continuing with simple fall detection mode")
            camera_config = None

    # Create detector based on user choice
    if args.detector == "mediapipe":
        detector = MediaPipePoseDetector()
        print("Using MediaPipe pose detector")
        fall_detector = MediaPipeFallDetector(camera_config=camera_config) if args.fall_detection else None
    else:
        detector = YOLOPoseDetector(args.model)
        print(f"Using YOLO pose detector with model: {args.model}")
        fall_detector = YOLOFallDetector(camera_config=camera_config) if args.fall_detection else None

    if args.fall_detection:
        if camera_config:
            print("Fall detection enabled with camera-aware multi-criteria analysis")
        else:
            print("Fall detection enabled (simple mode)")

    # Create frame provider based on input source
    playback_mode = PlaybackMode.PLAY if args.mode == "play" else PlaybackMode.STEP

    if args.video:
        if args.video.startswith("rtsp://"):
            frame_provider = RTSPFrameProvider(args.video)
            print(f"Processing RTSP stream: {args.video}")
        else:
            frame_provider = VideoFrameProvider(args.video, mode=playback_mode)
            print(f"Processing video: {args.video}")
    elif args.images:
        import os
        import glob
        
        # Expand directories to list of image files
        image_paths = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
        
        for path in args.images:
            if os.path.isdir(path):
                # It's a directory - load all image files from it
                print(f"DEBUG: '{path}' is a directory, scanning for images...")
                for ext in image_extensions:
                    pattern = os.path.join(path, f'*{ext}')
                    found_files = glob.glob(pattern)
                    print(f"DEBUG: Found {len(found_files)} files with extension {ext}")
                    image_paths.extend(found_files)
                    # Also check uppercase extensions
                    pattern = os.path.join(path, f'*{ext.upper()}')
                    found_files = glob.glob(pattern)
                    print(f"DEBUG: Found {len(found_files)} files with extension {ext.upper()}")
                    image_paths.extend(found_files)
            elif os.path.isfile(path):
                # It's a file - add it directly
                print(f"DEBUG: '{path}' is a file, adding to list")
                image_paths.append(path)
            else:
                print(f"WARNING: '{path}' is neither a file nor a directory, skipping")
        
        if not image_paths:
            print("ERROR: No valid image files found")
            return
        
        # Sort the paths for consistent ordering
        image_paths.sort()
        print(f"DEBUG: Total images to process: {len(image_paths)}")
        print(f"DEBUG: First few images: {image_paths[:5]}")
        
        frame_provider = ImageFrameProvider(image_paths, mode=playback_mode)
        print(f"Processing {len(image_paths)} images")
    else:
        # Default to camera
        camera_index = args.camera if args.camera is not None else 0
        frame_provider = CameraFrameProvider(camera_index)
        print(f"Using camera index: {camera_index}")

    # Create processor and run
    processor = PoseProcessor(
        detector,
        frame_provider,
        fall_detector,
        camera_config=camera_config,
        show_debug_info=args.debug,
    )
    processor.run()


if __name__ == "__main__":
    main()
