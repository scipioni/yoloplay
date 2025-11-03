# Camera Setup Guide

## Overview

This guide explains how to configure cameras for optimal fall detection performance using the enhanced camera-aware system.

## Camera Configuration Parameters

### Required Parameters

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| `height_meters` | Camera mounting height from floor | meters | 1.0 - 4.0 |
| `tilt_angle_degrees` | Camera tilt from horizontal plane | degrees | 0 - 90 |
| `horizontal_fov_degrees` | Horizontal field of view | degrees | 50 - 120 |
| `vertical_fov_degrees` | Vertical field of view | degrees | 40 - 90 |
| `image_width` | Frame width in pixels | pixels | 640 - 3840 |
| `image_height` | Frame height in pixels | pixels | 480 - 2160 |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `camera_id` | Unique identifier for the camera | auto-generated |
| `name` | Human-readable camera name | "Camera {id}" |
| `location` | Physical location description | "" |

## Configuration File Format

### YAML Format (`config/cameras.yaml`)

```yaml
cameras:
  entrance_camera:
    name: "Main Entrance Camera"
    location: "Building A - Main Entrance"
    height_meters: 2.8
    tilt_angle_degrees: 45
    horizontal_fov_degrees: 90
    vertical_fov_degrees: 60
    image_width: 1920
    image_height: 1080

  corridor_camera:
    name: "Corridor Camera"
    location: "Building A - 2nd Floor Corridor"
    height_meters: 2.5
    tilt_angle_degrees: 30
    horizontal_fov_degrees: 110
    vertical_fov_degrees: 70
    image_width: 1920
    image_height: 1080

  room_camera:
    name: "Patient Room Camera"
    location: "Room 201"
    height_meters: 2.2
    tilt_angle_degrees: 35
    horizontal_fov_degrees: 78
    vertical_fov_degrees: 52
    image_width: 1280
    image_height: 720
```

### JSON Format (`config/cameras.json`)

```json
{
  "cameras": {
    "entrance_camera": {
      "name": "Main Entrance Camera",
      "location": "Building A - Main Entrance",
      "height_meters": 2.8,
      "tilt_angle_degrees": 45,
      "horizontal_fov_degrees": 90,
      "vertical_fov_degrees": 60,
      "image_width": 1920,
      "image_height": 1080
    },
    "corridor_camera": {
      "name": "Corridor Camera",
      "location": "Building A - 2nd Floor Corridor",
      "height_meters": 2.5,
      "tilt_angle_degrees": 30,
      "horizontal_fov_degrees": 110,
      "vertical_fov_degrees": 70,
      "image_width": 1920,
      "image_height": 1080
    }
  }
}
```

## Common Camera Setups

### 1. Ceiling-Mounted Camera (High Angle)

**Best for**: Large open areas, hallways, entrances

```yaml
ceiling_camera:
  height_meters: 2.8
  tilt_angle_degrees: 60
  horizontal_fov_degrees: 90
  vertical_fov_degrees: 60
  image_width: 1920
  image_height: 1080
```

**Characteristics**:
- Wide coverage area
- Good for multiple person detection
- Requires strong perspective correction
- May have occlusion issues with furniture

**Recommended Settings**:
- Higher fall detection confidence threshold (0.75+)
- Enable temporal tracking for better accuracy
- Monitor for close-to-camera false positives

### 2. Eye-Level Camera (Horizontal View)

**Best for**: Doorways, narrow passages, specific monitoring zones

```yaml
eye_level_camera:
  height_meters: 1.6
  tilt_angle_degrees: 0
  horizontal_fov_degrees: 78
  vertical_fov_degrees: 52
  image_width: 1280
  image_height: 720
```

**Characteristics**:
- Natural viewing angle
- Minimal perspective distortion
- Limited vertical coverage
- Best for frontal fall detection

**Recommended Settings**:
- Standard confidence threshold (0.6)
- Aspect ratio criterion weighted higher
- Good for side-view fall detection

### 3. Wall-Mounted Camera (Medium Angle)

**Best for**: Patient rooms, offices, moderate-sized areas

```yaml
wall_camera:
  height_meters: 2.2
  tilt_angle_degrees: 30
  horizontal_fov_degrees: 110
  vertical_fov_degrees: 70
  image_width: 1920
  image_height: 1080
```

**Characteristics**:
- Balanced view between coverage and detail
- Moderate perspective correction needed
- Good compromise for most scenarios
- Suitable for furniture-filled spaces

**Recommended Settings**:
- Standard confidence threshold (0.65)
- Balanced multi-criteria weights
- Enable temporal tracking recommended

### 4. Corner-Mounted Camera (Diagonal View)

**Best for**: Room corners, comprehensive area coverage

```yaml
corner_camera:
  height_meters: 2.4
  tilt_angle_degrees: 40
  horizontal_fov_degrees: 120
  vertical_fov_degrees: 80
  image_width: 1920
  image_height: 1080
```

**Characteristics**:
- Maximum room coverage
- Strong perspective effects
- Good for fall trajectory analysis
- May require zone-based detection

**Recommended Settings**:
- Zone-specific thresholds
- High temporal tracking weight
- Distance-adaptive confidence thresholds

## Measuring Camera Parameters

### 1. Camera Height

**Tools needed**: Tape measure or laser distance meter

**Steps**:
1. Measure from floor to camera lens center
2. Record in meters (e.g., 2.8m, not 280cm)
3. Accuracy: ±5cm is acceptable

### 2. Tilt Angle

**Method 1: Using a smartphone app**
- Install angle measurement app (e.g., "Angle Meter")
- Align phone with camera viewing direction
- Read angle relative to horizontal

**Method 2: Using simple geometry**
- Measure: `h` = height difference between camera and target point
- Measure: `d` = horizontal distance to target point
- Calculate: `angle = arctan(h / d)` in degrees

**Method 3: Manufacturer specifications**
- Check camera documentation for mounting angle
- Verify with simple observation

### 3. Field of View (FOV)

**Method 1: Manufacturer specifications** (Recommended)
- Check camera datasheet
- Note both horizontal and vertical FOV
- If only diagonal FOV given, use aspect ratio to calculate:
  ```
  horizontal_fov = 2 * arctan(width / diagonal * tan(diagonal_fov / 2))
  vertical_fov = 2 * arctan(height / diagonal * tan(diagonal_fov / 2))
  ```

**Method 2: Practical measurement**
1. Place camera at known distance from wall (e.g., 3 meters)
2. Mark the edges of camera view on wall
3. Measure width and height of visible area
4. Calculate FOV:
   ```
   horizontal_fov = 2 * arctan(view_width / (2 * distance))
   vertical_fov = 2 * arctan(view_height / (2 * distance))
   ```

**Method 3: Using calibration software**
- OpenCV calibration tools
- Camera calibration apps
- Pro: Most accurate
- Con: Requires technical expertise

### 4. Image Resolution

**Simple method**:
- Capture a frame from the camera
- Check image properties (right-click > Properties)
- Note width × height in pixels

## Command-Line Usage

### Method 1: Using Configuration File

```bash
# Load configuration from YAML file, select specific camera
yoloplay --video rtsp://camera1/stream \
  --camera-config config/cameras.yaml \
  --camera-id entrance_camera \
  --fall-detection

# Load configuration from JSON file
yoloplay --video rtsp://camera2/stream \
  --camera-config config/cameras.json \
  --camera-id corridor_camera \
  --fall-detection
```

### Method 2: Inline Parameters

```bash
# Specify all parameters inline
yoloplay --video data/fall.webm \
  --camera-height 2.8 \
  --camera-tilt 45 \
  --camera-fov-h 90 \
  --camera-fov-v 60 \
  --fall-detection

# With detector selection
yoloplay --detector mediapipe \
  --video rtsp://camera/stream \
  --camera-height 2.2 \
  --camera-tilt 30 \
  --camera-fov-h 110 \
  --camera-fov-v 70 \
  --fall-detection
```

### Method 3: Mixed (File + Override)

```bash
# Load from file but override specific parameters
yoloplay --video rtsp://camera/stream \
  --camera-config config/cameras.yaml \
  --camera-id wall_camera \
  --camera-height 2.5 \  # Override height from file
  --fall-detection
```

### Method 4: No Configuration (Legacy Mode)

```bash
# Falls back to simple detection without camera parameters
yoloplay --video data/fall.webm --fall-detection
```

## Programmatic Usage

### Example 1: Load from File

```python
from yoloplay import YOLOPoseDetector, YOLOFallDetector, VideoFrameProvider, PoseProcessor
from yoloplay.camera_config import CameraConfig

# Load camera configuration
camera_config = CameraConfig.from_yaml("config/cameras.yaml", "entrance_camera")

# Create detector with camera config
detector = YOLOPoseDetector("yolov8n-pose.pt")
fall_detector = YOLOFallDetector(camera_config=camera_config)

# Create frame provider
frame_provider = VideoFrameProvider("data/fall.webm")

# Run processor
processor = PoseProcessor(detector, frame_provider, fall_detector)
processor.run()
```

### Example 2: Inline Configuration

```python
from yoloplay import YOLOPoseDetector, YOLOFallDetector, RTSPFrameProvider, PoseProcessor
from yoloplay.camera_config import CameraConfig

# Create camera configuration programmatically
camera_config = CameraConfig(
    height_meters=2.8,
    tilt_angle_degrees=45,
    horizontal_fov_degrees=90,
    vertical_fov_degrees=60,
    image_width=1920,
    image_height=1080,
    camera_id="entrance_camera",
    name="Main Entrance"
)

# Create components
detector = YOLOPoseDetector()
fall_detector = YOLOFallDetector(camera_config=camera_config)
frame_provider = RTSPFrameProvider("rtsp://camera/stream")

# Run
processor = PoseProcessor(detector, frame_provider, fall_detector)
processor.run()
```

### Example 3: Multi-Camera Setup

```python
from yoloplay import YOLOPoseDetector, YOLOFallDetector, RTSPFrameProvider, PoseProcessor
from yoloplay.camera_config import CameraConfigManager

# Load all camera configurations
config_manager = CameraConfigManager.from_yaml("config/cameras.yaml")

# Process multiple cameras
cameras = [
    ("rtsp://camera1/stream", "entrance_camera"),
    ("rtsp://camera2/stream", "corridor_camera"),
    ("rtsp://camera3/stream", "room_camera"),
]

for stream_url, camera_id in cameras:
    # Get camera-specific configuration
    camera_config = config_manager.get_camera(camera_id)
    
    # Create detector instances for this camera
    detector = YOLOPoseDetector()
    fall_detector = YOLOFallDetector(camera_config=camera_config)
    frame_provider = RTSPFrameProvider(stream_url)
    
    # Run in separate thread/process
    processor = PoseProcessor(detector, frame_provider, fall_detector)
    # ... threading/multiprocessing logic here
```

## Troubleshooting

### Problem: False Positives with Ceiling Camera

**Symptoms**: System detects falls when people sit or bend down

**Solution**:
- Increase `tilt_angle_degrees` to match actual camera angle
- Increase fall detection confidence threshold to 0.75+
- Enable temporal tracking to reduce single-frame false positives
- Consider adding a minimum fall duration threshold

### Problem: Missed Falls with Low Camera

**Symptoms**: System doesn't detect falls from eye-level cameras

**Solution**:
- Verify `height_meters` is accurately set
- Check that `tilt_angle_degrees` is close to 0 for eye-level
- Lower confidence threshold to 0.55-0.60
- Weight aspect ratio criterion higher in multi-criteria fusion

### Problem: Detection Varies by Distance

**Symptoms**: Falls detected when close but missed when far

**Solution**:
- Verify FOV parameters are correct
- Enable adaptive thresholding (should be automatic with camera config)
- Check that perspective correction is working
- Consider using higher resolution camera for distant subjects

### Problem: Inconsistent Detection Across Multiple Cameras

**Symptoms**: Some cameras work well, others don't

**Solution**:
- Verify each camera's configuration is accurate
- Measure and update camera parameters individually
- Check for lighting differences between cameras
- Ensure consistent detection model across all cameras
- Validate each camera's stream quality and frame rate

## Best Practices

### 1. Camera Placement

- **Height**: 2.0-3.0 meters ideal for most scenarios
- **Angle**: 30-45° for balanced coverage and accuracy
- **Coverage**: Ensure entire fall zone is visible
- **Lighting**: Adequate, consistent lighting critical
- **Obstacles**: Minimize furniture occlusion

### 2. Configuration Management

- **Document**: Keep detailed notes on each camera setup
- **Version control**: Track configuration file changes
- **Naming**: Use descriptive camera IDs (location-based)
- **Backup**: Maintain backup of working configurations
- **Testing**: Validate config with test videos before deployment

### 3. Performance Optimization

- **Resolution**: Use appropriate resolution for viewing distance
  - Near (<3m): 720p sufficient
  - Medium (3-6m): 1080p recommended
  - Far (>6m): 1080p or higher
- **Frame rate**: 15-30 FPS adequate for fall detection
- **Compression**: H.264/H.265 with moderate compression
- **Network**: Ensure stable network for RTSP streams

### 4. Maintenance

- **Regular validation**: Test system monthly with simulated falls
- **Parameter review**: Re-measure camera position if moved
- **Configuration updates**: Update config when cameras repositioned
- **Performance monitoring**: Track false positive/negative rates
- **Calibration checks**: Periodic validation of distance estimation

## Quick Start Checklist

- [ ] Measure camera height from floor
- [ ] Determine camera tilt angle
- [ ] Find camera FOV (check datasheet or measure)
- [ ] Note image resolution
- [ ] Create configuration entry in YAML/JSON
- [ ] Test with known fall video
- [ ] Adjust confidence threshold if needed
- [ ] Verify detection across different distances
- [ ] Document final configuration
- [ ] Deploy to production

## Additional Resources

- [Calibration Guide](calibration_guide.md) - Detailed calibration procedures
- [Fall Detection Plan](fall_detection_improvement_plan.md) - Architecture details
- [API Documentation](../README.md) - Programming interface
- Camera manufacturer documentation for technical specifications