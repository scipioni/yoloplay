import argparse
from typing import Dict, List, Optional


class Config:
    """Configuration class for the yoloplay application."""

    def __init__(self):
        self.detector: str = "yolo"
        self.model: str = "yolo11n-pose.pt"
        self.camera: Optional[int] = None
        self.video: Optional[str] = None
        self.images: Optional[List[str]] = None
        self.mode: str = "play"
        self.debug: bool = False
        self.calibrate: bool = False
        self.load_clusters: Optional[str] = None
        self.save: Optional[str] = None
        self.min_confidence: float = 0.55
        self.classifier: Optional[str] = None
        self.svm_model: Optional[str] = None
        self.svm_models: Optional[Dict[str, str]] = None  # Multiple SVM models: name -> path

    @classmethod
    def from_args(cls) -> "Config":
        """Parse command line arguments and return a Config instance."""
        parser = argparse.ArgumentParser(
            description="Pose detection with YOLO or MediaPipe"
        )
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
            # default="yolov8n-pose.pt",
            default="yolo11n-pose.pt",
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
            nargs="+",
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
            "--debug",
            action="store_true",
            help="Show detailed debug information and detection criteria",
        )
        parser.add_argument(
            "--calibrate",
            default="",
            help="Save calibration file",
        )
        parser.add_argument(
            "--load-clusters",
            type=str,
            help="Load cluster data from specified JSON file",
        )
        parser.add_argument(
            "--save",
            type=str,
            help="Save all keypoints to specified JSON file",
        )
        parser.add_argument(
            "--min-confidence",
            type=float,
            default=0.55,
            help="Minimum confidence threshold for filtering keypoints (default: 0.55)",
        )
        parser.add_argument(
            "--classifier",
            type=str,
            help="Path to trained classification model (.pt file) for keypoint classification",
        )
        parser.add_argument(
            "--svm-model",
            type=str,
            help="Path to trained SVM anomaly detection model (.pkl file)",
        )
        parser.add_argument(
            "--svm-models",
            type=str,
            nargs="+",
            help="Multiple SVM models in format 'name:path' (e.g., 'default:model1.pkl camera1:model2.pkl')",
        )

        args = parser.parse_args()

        config = cls()
        config.detector = args.detector
        config.model = args.model
        config.camera = args.camera
        config.video = args.video
        config.images = args.images
        config.mode = args.mode
        config.debug = args.debug
        config.calibrate = args.calibrate
        config.load_clusters = args.load_clusters
        config.save = args.save
        config.min_confidence = args.min_confidence
        config.classifier = args.classifier
        config.svm_model = args.svm_model

        # Parse multiple SVM models if provided
        if args.svm_models:
            config.svm_models = {}
            for model_spec in args.svm_models:
                if ":" not in model_spec:
                    parser.error(f"Invalid SVM model format: {model_spec}. Use 'name:path'")
                name, path = model_spec.split(":", 1)
                config.svm_models[name.strip()] = path.strip()

        return config


# Image file extensions supported by the application
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".gif",
}
