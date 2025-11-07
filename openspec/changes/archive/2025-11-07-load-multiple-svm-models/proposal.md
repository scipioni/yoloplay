# Change: Load Multiple SVM Models

## Why
Currently, the system can only load a single SVM anomaly detection model at a time. This limits the ability to use different models for different scenarios (e.g., different camera angles, lighting conditions, or pose types). Loading multiple models would enable more flexible and accurate anomaly detection by allowing the system to choose the most appropriate model based on context.

## What Changes
- Modify the `PoseProcessor` class to support loading and managing multiple SVM models
- Add configuration options to specify multiple model paths
- Update the anomaly detection logic to select and use the appropriate model using a label string with the name of the model
- Maintain backward compatibility with single model usage

## Impact
- Affected specs: svm-anomaly-detection
- Affected code: yoloplay/main.py (PoseProcessor class), yoloplay/config.py (configuration handling)
- No breaking changes to existing API