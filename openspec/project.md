# Project Context

## Purpose
Yoloplay is a flexible pose detection application that supports multiple pose detection models (YOLO and MediaPipe) with various input sources including real-time camera feeds, video files, RTSP streams, and image sequences. The primary goal is to provide real-time pose estimation and fall detection capabilities for computer vision applications, particularly in monitoring and safety scenarios.

Key features include:
- Multi-detector support (YOLO Pose via Ultralytics, MediaPipe Pose)
- Real-time fall detection using pose keypoints
- Flexible input sources (camera, video, RTSP, images)
- Playback controls for video and image processing
- SVM-based anomaly detection for pose classification
- Data collection and training pipeline for custom models

## Tech Stack
- **Language**: Python >=3.7
- **Core Libraries**:
  - OpenCV (computer vision and GUI)
  - NumPy (numerical computing)
  - PyTorch/TorchVision (deep learning framework)
  - Ultralytics YOLO (pose detection models)
  - MediaPipe (pose detection)
  - scikit-learn (machine learning, SVM)
  - PyYAML (configuration)
- **Build System**: setuptools with pyproject.toml
- **Package Manager**: uv (recommended for installation)
- **Development Tools**: pytest, black, flake8, mypy

## Project Conventions

### Code Style
- **Formatter**: Black (default settings)
- **Linter**: flake8
- **Type Checking**: mypy with strict mode
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Imports**: Standard library first, then third-party, then local modules
- **Docstrings**: Google-style docstrings for all public functions/classes

### Architecture Patterns
- **Abstract Base Classes**: Core abstractions for detectors (`PoseDetector`) and frame providers (`FrameProvider`)
- **Composition over Inheritance**: Main `PoseProcessor` class composes detector and frame provider
- **Separation of Concerns**: Detection logic separate from visualization and frame acquisition
- **Factory Pattern**: Command-line configuration creates appropriate detector/frame provider instances
- **Strategy Pattern**: Pluggable detectors and frame providers

### Testing Strategy
- **Framework**: pytest with pytest-cov for coverage reporting
- **Test Organization**: Unit tests for individual components, integration tests for full pipelines
- **Coverage Requirement**: Minimum 80% code coverage
- **CI/CD**: Automated testing on commits via GitHub Actions (planned)

### Git Workflow
- **Branching**: feature branches from main, squash merge
- **Commits**: Conventional commits format (type: description)
- **PR Reviews**: Required for all changes
- **Releases**: Versioned releases with changelog

## Domain Context
This project operates in computer vision and machine learning domain, specifically pose estimation and human activity recognition. Key concepts:

- **Pose Keypoints**: 17 standard COCO keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- **Confidence Scores**: Each keypoint has associated confidence from 0-1
- **Fall Detection**: Uses geometric relationships between keypoints to detect falls
- **Anomaly Detection**: SVM models trained on normal pose distributions to detect unusual poses
- **Normalization**: Keypoints normalized to image dimensions for consistent processing

AI assistants should understand OpenCV coordinate systems (top-left origin), keypoint indexing conventions, and the difference between absolute pixel coordinates and normalized coordinates.

## Important Constraints
- **Display Requirements**: GUI features require X11 display (OpenCV windows), but can run headless for processing-only tasks
- **Hardware**: GPU acceleration recommended for YOLO models, CPU fallback available
- **Memory**: YOLO models require significant RAM (>2GB for larger models)
- **Real-time Performance**: Target 30 FPS for real-time camera processing
- **Model Size**: Balance between accuracy and inference speed for deployment

## External Dependencies
- **Ultralytics YOLO**: Pose detection models (yolov8n-pose.pt, etc.)
- **MediaPipe**: Google's ML pipeline for pose detection
- **Pre-trained Models**: External model files downloaded on first use
- **Camera Hardware**: System cameras or RTSP streams for live input
- **Docker**: Optional containerization for deployment
