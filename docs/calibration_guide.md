# Camera Calibration Guide

## Overview

This guide provides detailed procedures for calibrating camera parameters to ensure accurate fall detection with camera height awareness. Proper calibration is essential for the perspective correction algorithms to work effectively.

## Calibration Methods Overview

| Method | Accuracy | Effort | Tools Required | Best For |
|--------|----------|--------|----------------|----------|
| **Manufacturer Data** | Medium | Low | Datasheet | Quick setup |
| **Physical Measurement** | High | Medium | Tape measure, app | Most scenarios |
| **Reference Object** | High | Medium | Known object, software | Precise setups |
| **Computer Vision** | Very High | High | OpenCV, expertise | Professional installations |

## Method 1: Using Manufacturer Specifications

### Advantages
- Fast and straightforward
- No additional tools needed
- Generally reliable for modern cameras

### Steps

1. **Locate Camera Documentation**
   - Find product datasheet or manual
   - Search manufacturer's website
   - Use model number to find specifications

2. **Extract Key Parameters**
   ```
   Look for:
   - Sensor size (e.g., 1/2.8", 1/3")
   - Focal length (e.g., 2.8mm, 3.6mm)
   - Field of View (FOV) - horizontal and vertical
   - Image resolution (e.g., 1920×1080)
   ```

3. **Calculate Missing FOV Values**
   
   If only diagonal FOV is provided:
   ```python
   import math
   
   def calculate_fov_from_diagonal(diagonal_fov_deg, aspect_ratio_w, aspect_ratio_h):
       """
       Calculate horizontal and vertical FOV from diagonal FOV.
       
       Args:
           diagonal_fov_deg: Diagonal field of view in degrees
           aspect_ratio_w: Width aspect ratio (e.g., 16 for 16:9)
           aspect_ratio_h: Height aspect ratio (e.g., 9 for 16:9)
       
       Returns:
           (horizontal_fov, vertical_fov) in degrees
       """
       diagonal_fov_rad = math.radians(diagonal_fov_deg)
       diagonal = math.sqrt(aspect_ratio_w**2 + aspect_ratio_h**2)
       
       horizontal_fov_rad = 2 * math.atan(
           (aspect_ratio_w / diagonal) * math.tan(diagonal_fov_rad / 2)
       )
       vertical_fov_rad = 2 * math.atan(
           (aspect_ratio_h / diagonal) * math.tan(diagonal_fov_rad / 2)
       )
       
       return (math.degrees(horizontal_fov_rad), math.degrees(vertical_fov_rad))
   
   # Example: 100° diagonal FOV, 16:9 aspect ratio
   h_fov, v_fov = calculate_fov_from_diagonal(100, 16, 9)
   print(f"Horizontal FOV: {h_fov:.1f}°")
   print(f"Vertical FOV: {v_fov:.1f}°")
   ```

4. **Measure Physical Parameters**
   - Camera mounting height: Use tape measure
   - Camera tilt angle: Use smartphone angle app
   - Verify installation matches intended configuration

### Limitations
- Manufacturer specs may be approximate
- Doesn't account for lens distortion
- May not reflect actual installation conditions

## Method 2: Physical Measurement with Known Distance

### Tools Required
- Tape measure or laser distance meter
- Smartphone with angle measurement app
- Optional: Level or inclinometer
- Masking tape for marking

### Procedure

#### Step 1: Measure Camera Height

1. **Identify Camera Lens Center**
   - Locate the center of the camera lens
   - This is your measurement reference point

2. **Measure Vertical Distance**
   ```
   Tools: Tape measure or laser distance meter
   
   From: Floor directly below camera
   To: Camera lens center
   
   Record: height_meters (e.g., 2.75m)
   Accuracy: ±5cm acceptable
   ```

3. **Document Measurement**
   ```yaml
   height_meters: 2.75  # Measured 2025-11-03
   ```

#### Step 2: Measure Tilt Angle

**Option A: Smartphone App Method**

1. Install angle measurement app:
   - iOS: "Clinometer", "Angle Meter"
   - Android: "Angle Meter", "Bubble Level"

2. Align phone with camera viewing direction:
   - Hold phone parallel to camera's optical axis
   - Use camera body or mounting bracket as reference
   - Take multiple readings and average

3. Record angle:
   ```yaml
   tilt_angle_degrees: 42  # Average of 3 readings: 41°, 42°, 43°
   ```

**Option B: Geometric Calculation**

1. Setup:
   - Place a target object at floor level
   - Center target in camera's field of view
   - Measure horizontal distance from camera to target

2. Measure:
   ```
   h = vertical distance from camera to target (height_meters)
   d = horizontal distance from camera to target (meters)
   ```

3. Calculate:
   ```python
   import math
   
   h = 2.75  # camera height in meters
   d = 3.5   # horizontal distance to target in meters
   
   tilt_angle_degrees = math.degrees(math.atan(h / d))
   print(f"Tilt angle: {tilt_angle_degrees:.1f}°")
   ```

#### Step 3: Measure Field of View

**Setup Requirements:**
- Clear wall or surface
- Sufficient space (3-5 meters from wall)
- Masking tape for marking
- Tape measure

**Procedure:**

1. **Position Camera**
   - Set camera at known distance from wall (e.g., 3.0 meters)
   - Align camera perpendicular to wall
   - Ensure camera is level (for accurate measurement)

2. **Mark Visible Area**
   - Display live camera feed
   - Mark edges of camera view on wall with tape:
     - Left edge
     - Right edge
     - Top edge
     - Bottom edge

3. **Measure Marked Area**
   ```
   view_width = distance between left and right marks (meters)
   view_height = distance between top and bottom marks (meters)
   distance_to_wall = 3.0 meters (or your measured distance)
   ```

4. **Calculate FOV**
   ```python
   import math
   
   def calculate_fov(view_dimension, distance):
       """Calculate field of view from physical measurements."""
       fov_radians = 2 * math.atan(view_dimension / (2 * distance))
       return math.degrees(fov_radians)
   
   # Example measurements
   view_width = 4.8      # meters
   view_height = 2.7     # meters
   distance = 3.0        # meters
   
   horizontal_fov = calculate_fov(view_width, distance)
   vertical_fov = calculate_fov(view_height, distance)
   
   print(f"Horizontal FOV: {horizontal_fov:.1f}°")
   print(f"Vertical FOV: {vertical_fov:.1f}°")
   ```

5. **Verify Calculation**
   - Typical camera FOV ranges: 50-120° horizontal
   - If result seems wrong, recheck measurements
   - Consider lens distortion for wide-angle cameras

#### Step 4: Record Image Resolution

1. **Capture Test Frame**
   ```bash
   # For RTSP stream
   ffmpeg -i rtsp://camera/stream -frames:v 1 test_frame.jpg
   
   # For local camera
   yoloplay --camera 0 --mode step
   # Press SPACE and check saved frame
   ```

2. **Check Image Properties**
   - Right-click image → Properties → Details
   - Or use command line:
     ```bash
     file test_frame.jpg
     # Output: JPEG image data, ... 1920 x 1080, ...
     ```

3. **Record Dimensions**
   ```yaml
   image_width: 1920
   image_height: 1080
   ```

## Method 3: Reference Object Calibration

### When to Use
- High accuracy requirements
- Precise distance estimation needed
- Professional installations
- Validating other methods

### Required
- Reference object of known dimensions (e.g., person of known height)
- Access to scene for object placement
- Image analysis software or manual measurement

### Procedure

#### Step 1: Prepare Reference Object

**Option A: Human Reference**
- Measure subject's height accurately (shoes off)
- Use standing height for vertical reference
- Shoulder width for horizontal reference
- Record measurements

**Option B: Physical Object**
- Use object with known dimensions:
  - Cardboard cutout (e.g., 1.7m × 0.5m "person")
  - Calibration chart
  - Ruler or measuring tape on wall
- Verify dimensions

#### Step 2: Capture Reference Frames

1. **Position Reference at Various Depths**
   ```
   Near position: Close to camera (1-2m)
   Mid position: Medium distance (3-4m)
   Far position: Far from camera (5-6m)
   ```

2. **Capture Frames**
   - Ensure reference is fully visible
   - Keep reference vertical and centered
   - Save frames for analysis

3. **Measure in Pixels**
   - Use image editor or OpenCV
   - Measure reference object height in pixels
   - Measure reference object width in pixels
   - Record for each position

#### Step 3: Calculate Camera Parameters

```python
import numpy as np
import math

def calibrate_from_reference(
    known_height_meters,
    pixel_heights,
    distances_meters,
    image_height_pixels
):
    """
    Calibrate vertical FOV from reference object measurements.
    
    Args:
        known_height_meters: Actual height of reference object
        pixel_heights: List of pixel heights at different distances
        distances_meters: List of distances to reference object
        image_height_pixels: Total image height in pixels
    
    Returns:
        Estimated vertical FOV in degrees
    """
    # Calculate angular height for each measurement
    angular_heights = []
    for pixel_h, dist in zip(pixel_heights, distances_meters):
        # Real-world height occupied at this distance
        real_h = known_height_meters
        # Angular height in radians
        ang_h = 2 * math.atan(real_h / (2 * dist))
        angular_heights.append(ang_h)
    
    # Average angular height per pixel
    avg_radians_per_pixel = np.mean([
        ang_h / pix_h for ang_h, pix_h in zip(angular_heights, pixel_heights)
    ])
    
    # Calculate total vertical FOV
    vertical_fov_rad = avg_radians_per_pixel * image_height_pixels
    vertical_fov_deg = math.degrees(vertical_fov_rad)
    
    return vertical_fov_deg

# Example usage
known_height = 1.75  # meters (person's height)
pixel_heights = [520, 350, 280]  # pixels at different distances
distances = [2.0, 3.0, 4.0]  # meters
image_height = 1080  # pixels

vfov = calibrate_from_reference(known_height, pixel_heights, distances, image_height)
print(f"Vertical FOV: {vfov:.1f}°")
```

#### Step 4: Validate Results

1. **Perform Consistency Check**
   ```python
   # Verify FOV across multiple measurements
   for pixel_h, dist in zip(pixel_heights, distances):
       expected_pixels = (known_height / dist) * (image_height / (2 * math.tan(math.radians(vfov / 2))))
       error_percent = abs(expected_pixels - pixel_h) / pixel_h * 100
       print(f"Distance {dist}m: {error_percent:.1f}% error")
   ```

2. **Accept if Error < 10%**
   - Larger errors indicate measurement issues
   - Re-measure if needed
   - Consider lens distortion for wide-angle

## Method 4: Computer Vision Calibration (Advanced)

### Overview
Uses OpenCV camera calibration for highest accuracy, accounting for lens distortion.

### Requirements
- Python with OpenCV (`pip install opencv-python`)
- Checkerboard calibration pattern
- Multiple calibration images
- Technical expertise

### Quick Procedure

1. **Print Calibration Pattern**
   - Download checkerboard pattern (e.g., 9×7 squares)
   - Print on rigid surface
   - Measure square size accurately

2. **Capture Calibration Images**
   ```python
   import cv2
   
   # Capture 20-30 images with pattern at different:
   # - Angles
   # - Distances
   # - Positions in frame
   
   cap = cv2.VideoCapture(0)
   for i in range(20):
       ret, frame = cap.read()
       cv2.imwrite(f'calib_{i:02d}.jpg', frame)
       # Move pattern, wait for enter
       input("Press enter for next capture...")
   ```

3. **Run Calibration**
   ```python
   import cv2
   import numpy as np
   import glob
   
   # Checkerboard dimensions (internal corners)
   chessboard_size = (9, 7)
   square_size = 0.025  # 25mm squares
   
   # Prepare object points
   objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
   objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
   objp *= square_size
   
   # Find corners in all images
   objpoints = []  # 3D points
   imgpoints = []  # 2D points
   
   images = glob.glob('calib_*.jpg')
   for fname in images:
       img = cv2.imread(fname)
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
       ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
       
       if ret:
           objpoints.append(objp)
           imgpoints.append(corners)
   
   # Calibrate camera
   ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
       objpoints, imgpoints, gray.shape[::-1], None, None
   )
   
   # Extract FOV from camera matrix
   fx = mtx[0, 0]  # focal length x
   fy = mtx[1, 1]  # focal length y
   width = gray.shape[1]
   height = gray.shape[0]
   
   horizontal_fov = 2 * math.degrees(math.atan(width / (2 * fx)))
   vertical_fov = 2 * math.degrees(math.atan(height / (2 * fy)))
   
   print(f"Horizontal FOV: {horizontal_fov:.1f}°")
   print(f"Vertical FOV: {vertical_fov:.1f}°")
   print(f"Distortion coefficients: {dist.ravel()}")
   ```

4. **Save Calibration Results**
   ```python
   # Save intrinsic parameters
   np.savez('camera_calibration.npz',
            mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
   ```

### Note on Lens Distortion
- Wide-angle cameras (>100° FOV) may have significant distortion
- Calibration provides distortion coefficients
- Consider undistorting images before fall detection
- Trade-off: Processing time vs. accuracy

## Validation and Testing

### Validation Checklist

After calibration, verify your configuration:

- [ ] **Sanity Check Values**
  - Height: 1.0m < height < 4.0m
  - Tilt: 0° ≤ tilt ≤ 90°
  - H-FOV: 50° < h_fov < 130°
  - V-FOV: 40° < v_fov < 100°
  - Resolution: Matches actual camera output

- [ ] **Distance Estimation Test**
  ```python
  # Place object at known distance
  # Verify estimated distance matches actual
  # Error should be < 20%
  ```

- [ ] **Fall Detection Test**
  ```bash
  # Test with known fall video
  yoloplay --video test_fall.mp4 \
    --camera-config config/cameras.yaml \
    --camera-id test_camera \
    --fall-detection
  ```

- [ ] **Cross-Validation**
  - Test with multiple methods
  - Results should agree within 10%
  - If discrepancies, investigate

### Common Issues and Solutions

**Issue**: FOV values seem too large (>120°)
- **Cause**: May have ultra-wide lens or measurement error
- **Solution**: Verify measurements, check for fisheye lens

**Issue**: Distance estimation wildly inaccurate
- **Cause**: Incorrect FOV or height parameters
- **Solution**: Re-calibrate, verify all measurements

**Issue**: Detection works at one distance but not others
- **Cause**: Fixed threshold without adaptive scaling
- **Solution**: Ensure camera config is properly loaded

**Issue**: Different results between YOLO and MediaPipe
- **Cause**: Different keypoint formats, coordinate systems
- **Solution**: Verify both detectors use same camera config

## Best Practices

### Measurement Tips

1. **Take Multiple Readings**
   - Measure 3 times, use average
   - Reduces random errors
   - Identifies measurement mistakes

2. **Document Everything**
   - Record measurement date
   - Note who performed calibration
   - Save test images/videos
   - Keep measurement worksheets

3. **Use Appropriate Tools**
   - Laser distance meter for long distances
   - Digital angle finder for precise angles
   - Quality tape measure (±1mm accuracy)

4. **Control Conditions**
   - Adequate lighting
   - Stable camera mounting
   - Calibrate in actual operating conditions

### Configuration Management

1. **Version Control**
   ```yaml
   # Include metadata in config
   entrance_camera:
     calibration_date: "2025-11-03"
     calibrated_by: "John Doe"
     calibration_method: "Physical measurement"
     validation_status: "Verified 2025-11-03"
     
     height_meters: 2.8
     tilt_angle_degrees: 45
     # ... other parameters
   ```

2. **Regular Re-Calibration**
   - After camera repositioning
   - Quarterly validation checks
   - After system upgrades
   - If detection accuracy degrades

3. **Backup Configurations**
   - Keep working configs in version control
   - Document any changes
   - Test before deploying to production

## Calibration Worksheet

Use this template for manual calibration:

```
Camera Calibration Worksheet
============================

Date: _______________
Calibrated by: _______________
Camera ID: _______________
Location: _______________

MEASUREMENTS
-----------
1. Camera Height
   From floor to lens center: _______ meters
   Measurement tool: _______________
   
2. Tilt Angle  
   Method: _______________
   Reading 1: _______ degrees
   Reading 2: _______ degrees
   Reading 3: _______ degrees
   Average: _______ degrees
   
3. Field of View (Method: _______________)
   
   For Physical Measurement:
   - Distance to wall: _______ meters
   - View width: _______ meters
   - View height: _______ meters
   - Calculated H-FOV: _______ degrees
   - Calculated V-FOV: _______ degrees
   
   For Manufacturer Data:
   - Model: _______________
   - Datasheet H-FOV: _______ degrees
   - Datasheet V-FOV: _______ degrees
   
4. Image Resolution
   Width: _______ pixels
   Height: _______ pixels

VALIDATION
----------
Test with fall video: _______________
Detection accuracy: [ ] Good  [ ] Fair  [ ] Poor
Distance estimation error: _______ %

NOTES
-----
_______________________________________________
_______________________________________________
_______________________________________________

CONFIGURATION ENTRY
-------------------
camera_id:
  name: "_______________"
  location: "_______________"
  height_meters: _______
  tilt_angle_degrees: _______
  horizontal_fov_degrees: _______
  vertical_fov_degrees: _______
  image_width: _______
  image_height: _______
  
  # Metadata
  calibration_date: "_______________"
  calibrated_by: "_______________"
  calibration_method: "_______________"
```

## Resources

### Recommended Apps

**iOS:**
- Clinometer + bubble level
- Angle Meter
- Ruler (for measurements)

**Android:**
- Bubble Level
- Angle Meter
- Smart Measure

**Desktop:**
- OpenCV Camera Calibration
- MATLAB Camera Calibrator
- ImageJ for pixel measurements

### Reference Materials

- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)
- [Understanding Camera FOV](https://en.wikipedia.org/wiki/Field_of_view)
- [Pinhole Camera Model](https://en.wikipedia.org/wiki/Pinhole_camera_model)

### Support

For calibration assistance:
- Check the [Camera Setup Guide](camera_setup_guide.md)
- Review the [Architectural Plan](fall_detection_improvement_plan.md)
- See [README](../README.md) for general usage