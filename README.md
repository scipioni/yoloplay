# yoloplay

A flexible pose detection application supporting both YOLO and MediaPipe detectors with various input sources (camera, video, images).

## Features

- **Multiple Pose Detectors:**
  - YOLO Pose models (via Ultralytics)
  - MediaPipe Pose detection
  - Extensible detector architecture for adding new detectors

- **Fall Detection:**
  - Real-time fall detection using pose keypoints
  - Support for both YOLO and MediaPipe detectors
  - Visual alerts and confidence scoring

- **Flexible Input Sources:**
  - Real-time camera input
  - Video file processing with play/pause/step controls
  - RTSP stream processing for live video feeds
  - Image file processing with navigation controls

- **Playback Controls:**
  - **Video mode:** Play/pause, step-by-step frame advance, mode toggle
  - **Image mode:** Navigate through images, auto-play or manual step-through
  - **Camera mode:** Real-time continuous capture

- **Clean Architecture:**
  - Abstract base classes for detectors and frame providers
  - Easy to extend with new detectors or input sources
  - Separation of concerns between detection, visualization, and frame acquisition

## Architecture

The application is built on two main class hierarchies:

### Detectors
- [`PoseDetector`](yoloplay/detectors.py:14) (abstract base class)
  - [`YOLOPoseDetector`](yoloplay/detectors.py:43) - YOLO-based pose detection
  - [`MediaPipePoseDetector`](yoloplay/detectors.py:129) - MediaPipe-based pose detection

### Frame Providers
- [`FrameProvider`](yoloplay/frame_providers.py:16) (abstract base class)
  - [`CameraFrameProvider`](yoloplay/frame_providers.py:50) - Real-time camera input
  - [`VideoFrameProvider`](yoloplay/frame_providers.py:70) - Video file with playback controls
  - [`RTSPFrameProvider`](yoloplay/frame_providers.py:149) - RTSP stream processing
  - [`ImageFrameProvider`](yoloplay/frame_providers.py:170) - Image sequence with navigation

## Installation

### Using Docker

1. Build the Docker image:
```bash
docker compose build
```

2. Run the application:
```bash
# Enable GUI support
xhost +local:docker

# Run with camera (default)
docker compose run --rm yoloplay yoloplay --video data/office.mkv

```

### Local Installation

```bash
uv pip install -e .
```

## Usage

### Command Line Options

```bash
yoloplay [OPTIONS]

Options:
  --detector {yolo,mediapipe}  Pose detector to use (default: yolo)
  --model PATH                 YOLO model path (default: yolov8n-pose.pt)
  --camera INDEX               Camera index for camera input
  --video PATH                 Path to video file
  --images PATH [PATH ...]     List of image files
  --mode {play,step}           Playback mode for video/images (default: play)
  --classifier PATH            Path to trained classification model (.pt file)
  --min-confidence FLOAT       Minimum confidence threshold for keypoints (default: 0.55)
  --debug                      Show detailed debug information
  --calibrate PATH             Save calibration data to specified file
  --load-clusters PATH         Load cluster data from specified file
  --save PATH                  Save keypoints to specified CSV file
```

### Examples

Create dataset for training:
```bash
yoloplay --video data/calibration.mkv --save data/train.csv
yolotrain --csv data/train.csv
yoloplay --video data/office.mkv --classifier  classification_best.pt
```

Use trained classifier for real-time pose classification:
```bash
# With video file
yoloplay --video data/office.mkv --classifier models/pose_classification_best.pt

# With camera
yoloplay --camera 0 --classifier models/pose_classification_best.pt

# With images
yoloplay --images data/fallen/*.jpg --classifier models/pose_classification_best.pt
```

The classifier will display the pose classification (standing/fallen) and confidence on the video feed in real-time.

Train pose classification model (with validation and early stopping):
```bash
# Basic training with defaults (80/20 split, patience=10)
yoloplay_train --csv data/train.csv --model-path models/pose_classification.pt

# Advanced training with custom parameters
yoloplay_train \
  --csv data/train.csv \
  --model-path models/pose_classification.pt \
  --epochs 200 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --split-ratio 0.7 \
  --patience 15 \
  --dropout 0.3 \
  --save-best-only
```

**Training Features:**
- **Train/Validation Split:** Automatically splits data (default 80/20) with stratified sampling
- **Early Stopping:** Stops training when validation loss stops improving (default patience: 10 epochs)
- **Best Model Checkpointing:** Saves the best model based on validation performance
- **Dropout Regularization:** Prevents overfitting with configurable dropout (default: 0.5)
- **Reproducibility:** Use `--random-seed` for consistent splits

**Training Arguments:**
- `--csv`: Path to CSV file with keypoints and labels (required)
- `--model-path`: Path to save trained model (default: classification.pt)
- `--epochs`: Maximum training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 0.001)
- `--device`: Device to use: 'auto', 'cpu', or 'cuda' (default: auto)
- `--split-ratio`: Train/val split ratio (default: 0.8)
- `--patience`: Early stopping patience in epochs (default: 10)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--dropout`: Dropout rate for regularization (default: 0.5)
- `--save-best-only`: Only save best model, not final model

**Training Output:**
```
Training on device: cuda
Loaded 1000 samples from CSV
Train set: 800 samples, Val set: 200 samples

Epoch 1/100
  Train - Loss: 0.6931, Acc: 0.5125
  Val   - Loss: 0.6823, Acc: 0.5300
  ✓ Best model saved (val_loss: 0.6823)

Epoch 2/100
  Train - Loss: 0.5234, Acc: 0.7450
  Val   - Loss: 0.5456, Acc: 0.7150
  ✓ Best model saved (val_loss: 0.5456)

...

Epoch 45/100
  Train - Loss: 0.1234, Acc: 0.9625
  Val   - Loss: 0.3456, Acc: 0.8650

Early stopping triggered at epoch 45
Best validation loss: 0.3256

Training completed!
Best model (epoch 35) saved to models/pose_classification_best.pt
  Train - Loss: 0.1456, Acc: 0.9500
  Val   - Loss: 0.3256, Acc: 0.8750
```

### Keyboard Controls

**Camera Mode:**
- `q` - Quit

**Video Mode:**
- `q` - Quit
- `p` - Toggle play/pause
- `SPACE` - Step to next frame
- `m` - Toggle between play and step modes

**Image Mode:**
- `q` - Quit
- `n` or `SPACE` - Next image
- `p` - Previous image
- `m` - Toggle between play and step modes

## Programmatic Usage

You can also use the library programmatically:

```python
from yoloplay import (
    YOLOPoseDetector,
    MediaPipePoseDetector,
    YOLOFallDetector,
    MediaPipeFallDetector,
    CameraFrameProvider,
    VideoFrameProvider,
    RTSPFrameProvider,
    ImageFrameProvider,
    PoseProcessor,
    PlaybackMode,
)

# Create a detector
detector = YOLOPoseDetector("yolov8n-pose.pt")
# or
detector = MediaPipePoseDetector()

# Create a frame provider
frame_provider = VideoFrameProvider("video.mp4", mode=PlaybackMode.STEP)
# or
frame_provider = RTSPFrameProvider("rtsp://example.com/stream")
# or
frame_provider = CameraFrameProvider(camera_index=0)
# or
frame_provider = ImageFrameProvider(["img1.jpg", "img2.jpg"], mode=PlaybackMode.PLAY)

# Create fall detector (optional)
fall_detector = YOLOFallDetector()  # or MediaPipeFallDetector()

# Create processor and run
processor = PoseProcessor(detector, frame_provider, fall_detector)
processor.run()
```
### Keypoint Classification

Use the trained model to classify pose keypoints:

```python
from yoloplay import KeypointClassifier
import numpy as np

# Load trained classifier
classifier = KeypointClassifier("models/pose_classification_best.pt")

# Single prediction
keypoints = np.array([...])  # 34 values: x1,y1,x2,y2,...,x17,y17
label, confidence = classifier(keypoints)
print(f"Prediction: {label} (confidence: {confidence:.2%})")
# Output: Prediction: standing (confidence: 95.23%)

# Batch prediction
keypoints_batch = np.array([...])  # Shape: (N, 34)
results = classifier.predict_batch(keypoints_batch)
for i, (label, conf) in enumerate(results):
    print(f"Sample {i}: {label} ({conf:.2%})")
```

**KeypointClassifier API:**
- `predict(keypoints)`: Classify single keypoint array (34 values)
- `predict_batch(keypoints_batch)`: Classify multiple keypoint arrays  
- `__call__(keypoints)`: Shorthand for `predict()`

**Returns:**
- `label`: 'standing' or 'fallen'
- `confidence`: Probability score (0.0 to 1.0)


## Extending the Application

### Adding a New Detector

Create a new class that inherits from [`PoseDetector`](yoloplay/detectors.py:14):

```python
from yoloplay.detectors import PoseDetector

class MyCustomDetector(PoseDetector):
    def detect(self, frame):
        # Your detection logic
        pass
    
    def visualize(self, frame, results):
        # Your visualization logic
        pass
```

### Adding a New Fall Detector

Create a new class that inherits from [`FallDetector`](yoloplay/fall_detector.py:8):

```python
from yoloplay.fall_detector import FallDetector

class MyCustomFallDetector(FallDetector):
    def detect_fall(self, keypoints):
        # Your custom fall detection logic
        # Return (is_fallen, confidence)
        pass
```

### Adding a New Frame Provider

Create a new class that inherits from [`FrameProvider`](yoloplay/frame_providers.py:16):

```python
from yoloplay.frame_providers import FrameProvider

class MyCustomProvider(FrameProvider):
    def open(self):
        # Open your source
        pass
    
    def read(self):
        # Read next frame
        pass
    
    def release(self):
        # Release resources
        pass
```

### Notes

register video
```
mpv rtsp://10.1.109.144:554/s0 --stream-record=office.mkv
``` 