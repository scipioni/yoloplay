# yoloplay

This package provides functionality to use yolopose. 

<!-- In particularly useful for applications that need to transform pose coordinates from a perspective view (e.g., from a camera at an angle to the floor) to a front-facing view where floor measurements become accurate. -->

## Features

<!-- - 4-point floor calibration for accurate perspective transformation
- Support for mapping YOLO Pose COCO format keypoints
- Configurable transformation parameters
- Visualization of calibration points and results
- Saving of transformation matrices for consistent mapping
- Live camera support for real-time processing
- Calibration mode for setting up the 4-point transformation
- Camera height configuration (150cm default)
- Fallen person detection with visual effects on bones (red color) -->


## Usage

1. Build the Docker image:
```bash
docker compose build
```


```bash
# Run continuously in the background (with GUI)
xhost +local:docker
docker compose run --rm yoloplay bash

```

