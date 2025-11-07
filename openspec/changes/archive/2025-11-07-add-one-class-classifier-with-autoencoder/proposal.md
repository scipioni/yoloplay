# Change: Add One-Class Classifier with Autoencoder

## Why
The current SVM-based anomaly detection is limited to binary classification and may not capture complex pose patterns effectively. Adding a one-class classifier using autoencoders will provide unsupervised anomaly detection that can learn normal pose distributions more robustly, potentially improving fall detection accuracy in diverse environments.

## What Changes
- Add new capability: `one-class-classifier` for autoencoder-based anomaly detection
- Implement autoencoder architecture for pose keypoints reconstruction
- Integrate one-class SVM on autoencoder latent space for anomaly scoring
- Add configuration support for autoencoder parameters (latent dimensions, layers, etc.) and set defaults to best learn vectors of keypoints (17 points of x,y)
- Extend detector interface to support multiple anomaly detection methods

## Impact
- Affected specs: New capability `one-class-classifier`
- Affected code: `yoloplay/detectors.py`, `yoloplay/config.py`, new `yoloplay/autoencoder.py`
- No breaking changes to existing SVM functionality
- Use libraries specified in pyproject.toml