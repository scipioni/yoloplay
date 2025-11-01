# YOLO Pose Vertical Mapper

This package provides functionality to map YOLO Pose coordinates from an original image plane to a vertical plane using 4-point floor calibration. This is particularly useful for applications that need to transform pose coordinates from a perspective view (e.g., from a camera at an angle to the floor) to a front-facing view where floor measurements become accurate.

## Features

- 4-point floor calibration for accurate perspective transformation
- Support for mapping YOLO Pose COCO format keypoints
- Configurable transformation parameters
- Visualization of calibration points and results
- Saving of transformation matrices for consistent mapping
- Live camera support for real-time processing
- Calibration mode for setting up the 4-point transformation
- Docker support with CUDA 12 for GPU acceleration
- Camera height configuration (150cm default)
- Fallen person detection with visual effects on bones (red color)

## Installation

### Standard Installation

To install this package, you can use pip after navigating to the project directory:

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

### Docker Installation (Recommended for GPU acceleration)

If you have a CUDA-compatible GPU and nvidia-container-toolkit installed, you can use the Docker setup:

```bash
# Build and run with Docker Compose (for batch processing)
docker-compose up --build yolopose_mapper

# Build and run with Docker Compose (for live camera mode)
docker-compose up --build yolopose_mapper_camera
```

## Configuration

The `config.yaml` file contains all the important parameters for the transformation:

- `calibration_points`: Define the 4 points in the original image that form the floor area
- `target_points`: Define where these points should map to in the vertical plane
- `transformation`: Parameters for the type of transformation (perspective or affine)
- `output`: Output image dimensions and transformation matrix saving options
- `yolo_pose`: Keypoint indices and confidence thresholds
- `other`: Visualization and other parameters

## Usage

### Batch Mode (Processing Static Images)

```bash
python -m yolopose_vertical_mapper.main --mode batch --config config.yaml --image input.jpg --keypoints keypoints.json --output_image output.jpg --output_keypoints transformed_keypoints.json --visualize
```

### Live Camera Mode

```bash
# Process live camera feed
python -m yolopose_vertical_mapper.main --mode live --config config.yaml --camera 0

# Run in calibration mode to set up 4-point calibration
python -m yolopose_vertical_mapper.main --mode live --config config.yaml --calibrate

# Process with only transformed view (no original)
python -m yolopose_vertical_mapper.main --mode live --config config.yaml --no-display-original
```

### Calibration Mode

The enhanced calibration mode allows you to visually set up the 4-point floor calibration:

1. Run the calibration mode:
   ```bash
   python -m yolopose_vertical_mapper.main --mode live --config config.yaml --calibrate
   ```

2. A live camera window will appear. Click on 4 floor points in this order:
   - Point 1: Top-left corner of your floor area
   - Point 2: Top-right corner of your floor area
   - Point 3: Bottom-right corner of your floor area
   - Point 4: Bottom-left corner of your floor area

3. After clicking all 4 points, press 'c' to capture and automatically save the calibration points to your config.yaml file.

4. Press 'q' to quit the calibration mode.

The calibration points will be saved directly to your configuration file. The target points are automatically set to a 400x300 rectangle, which you can modify in the config file later if needed.

Note: The calibration points are saved to the config file automatically when you press 'c', so you don't need to manually copy them.

### Docker Usage

```bash
# Build the Docker image
docker-compose build yolopose_mapper

# Run batch processing in Docker
docker-compose run --rm yolopose_mapper --mode batch --config /workspace/yolopose_vertical_mapper/config.yaml --image /workspace/data/input.jpg --keypoints /workspace/data/keypoints.json

# Run live camera mode in Docker
docker-compose run --rm yolopose_mapper_camera python -m yolopose_vertical_mapper.main --mode live --config /workspace/yolopose_vertical_mapper/config.yaml --camera 0

# Run calibration mode in Docker (automatically saves to config)
docker-compose run --rm yolopose_mapper_camera python -m yolopose_vertical_mapper.main --mode live --config /workspace/yolopose_vertical_mapper/config.yaml --calibrate

# Or use docker-compose up to run as a service
docker-compose up yolopose_mapper_camera
```

### Docker Calibration Mode

For live camera calibration with automatic config saving:

```bash
# Run calibration mode with camera access
docker-compose run --rm yolopose_mapper_camera python -m yolopose_vertical_mapper.main --mode live --config /workspace/yolopose_vertical_mapper/config.yaml --calibrate

# Follow the same steps as above:
# 1. Click on 4 floor points in order (top-left, top-right, bottom-right, bottom-left)
# 2. Press 'c' to capture and automatically save points to config file
# 3. Press 'q' to quit
```

### Fallen Person Detection

The system includes intelligent fallen person detection based on pose analysis:

- **Torso angle analysis**: Calculates the angle between the torso and ground plane
- **Threshold-based detection**: Persons with torso angles below 30° are considered fallen
- **Visual indicators**: Fallen persons are highlighted with red bones and keypoints
- **Configurable parameters**: Adjust sensitivity through the config.yaml file

The detection algorithm analyzes the pose keypoints to determine if a person is lying on the ground, making it ideal for elderly care or safety monitoring applications.

### X11 Display Issues

### Fallen Person Detection

The system includes intelligent fallen person detection based on pose analysis:

- **Torso angle analysis**: Calculates the angle between the torso and ground plane
- **Threshold-based detection**: Persons with torso angles below 30° are considered fallen
- **Visual indicators**: Fallen persons are highlighted with red bones and keypoints
- **Configurable parameters**: Adjust sensitivity through the config.yaml file

The detection algorithm analyzes the pose keypoints to determine if a person is lying on the ground, making it ideal for elderly care or safety monitoring applications.

### Alternative Camera Entry Point

```bash
# Use the dedicated camera entry point
python -m yolopose_vertical_mapper.camera_handler --config config.yaml --camera 0
```

### Programmatic Usage

```python
from yolopose_vertical_mapper.vertical_mapper import YoloPoseVerticalMapper
from yolopose_vertical_mapper.camera_handler import CameraPoseProcessor
import numpy as np
import cv2

# For batch processing:
mapper = YoloPoseVerticalMapper('./config.yaml')
image = cv2.imread('input.jpg')
keypoints = np.load('keypoints.npy')  # Shape: (N, 17, 3) for N people
transformed_keypoints = mapper.map_pose_keypoints(keypoints)
transformed_image = mapper.apply_to_image(image)

# For live camera processing:
processor = CameraPoseProcessor('./config.yaml')
processor.run_camera_loop(camera_index=0)  # Use default camera
```

## Example

An example usage script is provided in `example_usage.py` that demonstrates the functionality with sample data.

## Requirements

### For local installation:
- Python >= 3.7
- numpy
- opencv-python
- pyyaml
- torch
- torchvision
- ultralytics

### For Docker installation:
- Docker
- nvidia-container-toolkit
- CUDA-compatible GPU

## Docker Setup

### Prerequisites

1. Install Docker: https://docs.docker.com/get-docker/
2. Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

### Usage

1. Build the Docker image:
```bash
docker-compose build yolopose_mapper
```

2. For batch processing:
```bash
# Place your input files in the ./data directory
mkdir -p data
cp your_input.jpg data/
cp your_keypoints.json data/

# Run batch processing
docker-compose run --rm yolopose_mapper --mode batch --config /workspace/yolopose_vertical_mapper/config.yaml --image /workspace/data/your_input.jpg --keypoints /workspace/data/your_keypoints.json --output_image /workspace/data/output.jpg --output_keypoints /workspace/data/transformed_keypoints.json
```

3. For live camera mode with GUI:
```bash
# Allow Docker to access X11 display (run once per session)
xhost +local:docker

# Run live camera processing (with X11 forwarding for GUI)
docker-compose run --rm --device /dev/video0 yolopose_mapper_camera
```

4. For live camera mode without GUI (headless):
```bash
# Run without GUI display
docker-compose run --rm --device /dev/video0 yolopose_mapper_camera --no-display-original
```

5. Or run as a service:
```bash
# Run continuously in the background (with GUI)
xhost +local:docker
docker-compose up yolopose_mapper_camera

# Or run continuously without GUI
docker-compose up --service-ports yolopose_mapper_camera
```

### X11 Display Issues

If you encounter Qt display errors like:
```
qt.qpa.xcb: could not connect to display :0
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```

This means the container cannot access the X11 display. To fix this:

1. Make sure you run `xhost +local:docker` before starting the container
2. Or run in headless mode using the `--no-display-original` flag
3. If you're running on a headless server, avoid using GUI features

Note that the calibration mode works with or without X11 display access. You can run it on a headless server as long as you have camera access.

The Docker setup mounts the package source code at runtime, so any changes to your local code will be reflected in the container without rebuilding.

## License

MIT