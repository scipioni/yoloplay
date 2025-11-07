# One-Class Classifier with Autoencoder

## ADDED Requirements

### Requirement: Autoencoder Architecture
The system SHALL implement an autoencoder neural network for unsupervised learning of normal pose keypoint distributions. The autoencoder SHALL consist of an encoder that compresses pose keypoints into a latent representation and a decoder that reconstructs the original keypoints from the latent space.

#### Scenario: Pose Keypoint Encoding
- **WHEN** pose keypoints are provided as input
- **THEN** the encoder compresses the 17 COCO keypoints into a lower-dimensional latent vector
- **AND** the latent representation captures essential pose structure

#### Scenario: Pose Reconstruction
- **WHEN** a latent vector is provided
- **THEN** the decoder reconstructs the original pose keypoints
- **AND** reconstruction error indicates anomaly likelihood

### Requirement: One-Class SVM Integration
The system SHALL integrate one-class SVM on the autoencoder's latent space for anomaly detection. The SVM SHALL be trained on latent representations of normal poses to establish a decision boundary.

#### Scenario: Latent Space Classification
- **WHEN** a pose is encoded to latent space
- **THEN** the one-class SVM scores the latent vector for normality
- **AND** scores below threshold indicate anomalous poses

#### Scenario: Training on Normal Data
- **WHEN** normal pose data is available
- **THEN** the autoencoder is trained to minimize reconstruction error
- **AND** the one-class SVM is trained on resulting latent vectors

### Requirement: Configuration Support
The system SHALL support configuration of autoencoder parameters including latent dimensions, encoder/decoder layer sizes, learning rate, and training epochs.

#### Scenario: Autoencoder Configuration
- **WHEN** autoencoder parameters are specified in configuration
- **THEN** the model architecture adapts to the specified dimensions
- **AND** training parameters are applied during model training

### Requirement: Anomaly Scoring
The system SHALL provide anomaly scores combining reconstruction error and one-class SVM decision function. Scores SHALL be normalized to [0,1] range where higher values indicate higher anomaly likelihood.

#### Scenario: Combined Scoring
- **WHEN** a pose is processed
- **THEN** reconstruction error and SVM score are combined
- **AND** normalized anomaly score is returned for fall detection

#### Scenario: Threshold-Based Detection
- **WHEN** anomaly score exceeds configured threshold
- **THEN** the pose is classified as anomalous
- **AND** fall detection logic is triggered