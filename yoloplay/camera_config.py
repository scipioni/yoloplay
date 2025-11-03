"""
Camera configuration module for perspective-aware fall detection.

This module provides classes and utilities for managing camera-specific parameters
such as mounting height, tilt angle, and field of view, which are essential for
accurate fall detection across different camera setups.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class CameraConfig:
    """
    Camera configuration data structure.
    
    Stores camera-specific parameters for perspective-aware fall detection.
    All measurements should be in standard units (meters, degrees, pixels).
    
    Attributes:
        height_meters: Camera mounting height from ground (meters)
        tilt_angle_degrees: Camera tilt angle from horizontal plane (0-90°)
        horizontal_fov_degrees: Horizontal field of view (degrees)
        vertical_fov_degrees: Vertical field of view (degrees)
        image_width: Frame width in pixels
        image_height: Frame height in pixels
        camera_id: Unique identifier for the camera
        name: Human-readable camera name
        location: Physical location description
    """
    
    height_meters: float
    tilt_angle_degrees: float
    horizontal_fov_degrees: float
    vertical_fov_degrees: float
    image_width: int
    image_height: int
    camera_id: str = "default"
    name: str = ""
    location: str = ""
    
    # Optional metadata
    calibration_date: str = ""
    calibrated_by: str = ""
    calibration_method: str = ""
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self.validate()
        
        # Set default name if not provided
        if not self.name:
            self.name = f"Camera {self.camera_id}"
    
    def validate(self) -> None:
        """
        Validate camera configuration parameters.
        
        Raises:
            ValueError: If any parameter is out of valid range
        """
        # Height validation
        if not (0.5 <= self.height_meters <= 10.0):
            raise ValueError(
                f"Camera height must be between 0.5 and 10.0 meters, got {self.height_meters}"
            )
        
        # Tilt angle validation
        if not (0 <= self.tilt_angle_degrees <= 90):
            raise ValueError(
                f"Tilt angle must be between 0 and 90 degrees, got {self.tilt_angle_degrees}"
            )
        
        # FOV validation
        if not (10 <= self.horizontal_fov_degrees <= 180):
            raise ValueError(
                f"Horizontal FOV must be between 10 and 180 degrees, got {self.horizontal_fov_degrees}"
            )
        
        if not (10 <= self.vertical_fov_degrees <= 180):
            raise ValueError(
                f"Vertical FOV must be between 10 and 180 degrees, got {self.vertical_fov_degrees}"
            )
        
        # Image resolution validation
        if not (320 <= self.image_width <= 7680):
            raise ValueError(
                f"Image width must be between 320 and 7680 pixels, got {self.image_width}"
            )
        
        if not (240 <= self.image_height <= 4320):
            raise ValueError(
                f"Image height must be between 240 and 4320 pixels, got {self.image_height}"
            )
    
    def get_focal_length_pixels(self) -> tuple[float, float]:
        """
        Calculate focal length in pixels from FOV.
        
        Returns:
            Tuple of (focal_length_x, focal_length_y) in pixels
        """
        fx = self.image_width / (2 * math.tan(math.radians(self.horizontal_fov_degrees) / 2))
        fy = self.image_height / (2 * math.tan(math.radians(self.vertical_fov_degrees) / 2))
        return fx, fy
    
    def get_camera_matrix(self):
        """
        Get camera intrinsic matrix (3x3).
        
        Returns:
            3x3 camera matrix as nested list [[...], [...], [...]]
        """
        fx, fy = self.get_focal_length_pixels()
        cx = self.image_width / 2
        cy = self.image_height / 2
        
        return [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    
    @classmethod
    def from_dict(cls, data: dict) -> "CameraConfig":
        """
        Create CameraConfig from dictionary.
        
        Args:
            data: Dictionary containing camera configuration
            
        Returns:
            CameraConfig instance
        """
        return cls(**data)
    
    def to_dict(self) -> dict:
        """
        Convert CameraConfig to dictionary.
        
        Returns:
            Dictionary representation of camera configuration
        """
        return {
            "height_meters": self.height_meters,
            "tilt_angle_degrees": self.tilt_angle_degrees,
            "horizontal_fov_degrees": self.horizontal_fov_degrees,
            "vertical_fov_degrees": self.vertical_fov_degrees,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_id": self.camera_id,
            "name": self.name,
            "location": self.location,
            "calibration_date": self.calibration_date,
            "calibrated_by": self.calibrated_by,
            "calibration_method": self.calibration_method,
        }


class CameraConfigManager:
    """
    Manages multiple camera configurations.
    
    Provides utilities for loading, saving, and accessing camera configurations
    for multi-camera setups.
    """
    
    def __init__(self):
        """Initialize empty camera configuration manager."""
        self.cameras: Dict[str, CameraConfig] = {}
    
    def add_camera(self, config: CameraConfig) -> None:
        """
        Add a camera configuration.
        
        Args:
            config: CameraConfig instance to add
        """
        self.cameras[config.camera_id] = config
    
    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        """
        Get camera configuration by ID.
        
        Args:
            camera_id: Unique camera identifier
            
        Returns:
            CameraConfig instance or None if not found
        """
        return self.cameras.get(camera_id)
    
    def list_cameras(self) -> list[str]:
        """
        Get list of all camera IDs.
        
        Returns:
            List of camera IDs
        """
        return list(self.cameras.keys())
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "CameraConfigManager":
        """
        Load camera configurations from YAML file.
        
        Args:
            filepath: Path to YAML configuration file
            
        Returns:
            CameraConfigManager instance with loaded configurations
            
        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML structure is invalid
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML config loading. "
                "Install it with: pip install pyyaml"
            )
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict) or 'cameras' not in data:
            raise ValueError("YAML file must contain a 'cameras' key at root level")
        
        manager = cls()
        
        for camera_id, camera_data in data['cameras'].items():
            camera_data['camera_id'] = camera_id
            config = CameraConfig.from_dict(camera_data)
            manager.add_camera(config)
        
        return manager
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "CameraConfigManager":
        """
        Load camera configurations from JSON file.
        
        Args:
            filepath: Path to JSON configuration file
            
        Returns:
            CameraConfigManager instance with loaded configurations
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON structure is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or 'cameras' not in data:
            raise ValueError("JSON file must contain a 'cameras' key at root level")
        
        manager = cls()
        
        for camera_id, camera_data in data['cameras'].items():
            camera_data['camera_id'] = camera_id
            config = CameraConfig.from_dict(camera_data)
            manager.add_camera(config)
        
        return manager
    
    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """
        Save camera configurations to YAML file.
        
        Args:
            filepath: Path to output YAML file
            
        Raises:
            ImportError: If PyYAML is not installed
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML config saving. "
                "Install it with: pip install pyyaml"
            )
        
        data = {
            'cameras': {
                camera_id: config.to_dict()
                for camera_id, config in self.cameras.items()
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, filepath: Union[str, Path], indent: int = 2) -> None:
        """
        Save camera configurations to JSON file.
        
        Args:
            filepath: Path to output JSON file
            indent: Indentation level for pretty printing
        """
        data = {
            'cameras': {
                camera_id: config.to_dict()
                for camera_id, config in self.cameras.items()
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)


def load_camera_config(
    filepath: Optional[Union[str, Path]] = None,
    camera_id: Optional[str] = None,
    **kwargs
) -> Optional[CameraConfig]:
    """
    Convenience function to load a single camera configuration.
    
    Args:
        filepath: Path to configuration file (YAML or JSON)
        camera_id: Camera ID to load from file
        **kwargs: Direct camera parameters (if no file provided)
        
    Returns:
        CameraConfig instance or None
        
    Examples:
        # Load from file
        config = load_camera_config("config/cameras.yaml", "entrance_camera")
        
        # Create from parameters
        config = load_camera_config(
            height_meters=2.8,
            tilt_angle_degrees=45,
            horizontal_fov_degrees=90,
            vertical_fov_degrees=60,
            image_width=1920,
            image_height=1080
        )
    """
    if filepath:
        filepath = Path(filepath)
        
        if filepath.suffix in ['.yaml', '.yml']:
            manager = CameraConfigManager.from_yaml(filepath)
        elif filepath.suffix == '.json':
            manager = CameraConfigManager.from_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        if camera_id:
            return manager.get_camera(camera_id)
        elif len(manager.cameras) == 1:
            # Return the only camera if just one exists
            return next(iter(manager.cameras.values()))
        else:
            raise ValueError(
                f"Multiple cameras found in {filepath}. "
                f"Please specify camera_id from: {manager.list_cameras()}"
            )
    
    elif kwargs:
        # Create config from parameters
        return CameraConfig(**kwargs)
    
    return None