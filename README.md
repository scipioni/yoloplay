# yoloplay

A flexible pose detection application supporting both YOLO and MediaPipe detectors with various input sources (camera, video, images).

## Features

- **Multiple Pose Detectors:**
  - YOLO Pose models (via Ultralytics)
  - MediaPipe Pose detection
  - Extensible detector architecture for adding new detectors

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
docker compose run --rm yoloplay python -m yoloplay.main

# Run with video file
docker compose run --rm yoloplay python -m yoloplay.main --video data/fall.webm

# Run with RTSP stream
docker compose run --rm yoloplay python -m yoloplay.main --video rtsp://example.com/stream

# Run with MediaPipe detector
docker compose run --rm yoloplay python -m yoloplay.main --detector mediapipe
```

### Local Installation

```bash
pip install -e .
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
```

### Examples

**Camera with YOLO detector:**
```bash
yoloplay --camera 0
```

**Video with step-through mode:**
```bash
yoloplay --video data/fall.webm --mode step
```

**Images with MediaPipe detector:**
```bash
yoloplay --detector mediapipe --images img1.jpg img2.jpg img3.jpg
```

**Video with YOLO and auto-play mode:**
```bash
yoloplay --video data/fall.webm --mode play
```

**RTSP stream with MediaPipe detector:**
```bash
yoloplay --detector mediapipe --video rtsp://192.168.1.100:554/stream
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

# Create processor and run
processor = PoseProcessor(detector, frame_provider)
processor.run()
```

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

## License

MIT
