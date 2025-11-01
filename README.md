# yoloplay

This package provides functionality to use yolopose with enhanced pose visualization and configurable camera parameters.

## Features

- Real-time pose estimation using YOLO Pose models
- Visualization of pose keypoints and skeletal connections (bones)
- Configurable camera height parameter for accurate floor measurements
- Live camera support for real-time processing
- Dockerized deployment for easy setup and portability


## Usage

1. Build the Docker image:
```bash
docker compose build
```

2. Run the application:
```bash
# Run continuously in the background (with GUI)
xhost +local:docker
docker compose run --rm yoloplay python -m yoloplay.main

# Run with a specific camera height (in cm)
docker compose run --rm yoloplay python -m yoloplay.main --height 150

# For other options, use the help command
docker compose run --rm yoloplay python -m yoloplay.main --help
```

