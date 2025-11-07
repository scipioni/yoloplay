# Change: Add K-Means Classifier

## Why
The project currently supports SVM-based anomaly detection and one-class classification with autoencoders. Adding K-Means clustering as a classifier provides an additional unsupervised learning approach for pose classification, offering complementary capabilities for grouping similar poses and detecting anomalies based on cluster distances.

## What Changes
- Add new K-Means classifier capability for pose clustering and anomaly detection
- Integrate K-Means into the existing detector framework
- Provide configuration options for number of clusters and distance thresholds
- Add training and inference methods for K-Means models

## Impact
- Affected specs: k-means-classifier (new capability)
- Affected code: yoloplay/detectors.py, yoloplay/config.py, yoloplay/main.py
- No breaking changes to existing functionality