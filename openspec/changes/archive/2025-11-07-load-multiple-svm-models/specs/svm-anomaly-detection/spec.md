## MODIFIED Requirements
### Requirement: SVM Model Loading
The system SHALL support loading multiple SVM anomaly detection models simultaneously. Each model SHALL be identified by a unique name and associated with specific detection criteria.

#### Scenario: Load Multiple Models
- **WHEN** multiple SVM model paths are provided in configuration
- **THEN** all models are loaded and stored with their identifiers
- **AND** each model can be accessed by its name

#### Scenario: Model Selection by Name
- **WHEN** keypoints are processed for anomaly detection
- **THEN** the system selects the most appropriate SVM model based on its name
- **AND** falls back to a default model if no specific criteria match

#### Scenario: Backward Compatibility
- **WHEN** only a single SVM model path is provided
- **THEN** the system operates identically to the previous single-model implementation
- **AND** no additional configuration is required

## ADDED Requirements
### Requirement: Model Configuration
The system SHALL accept configuration specifying multiple SVM models with their selection criteria. Configuration SHALL include model paths, names, and optional selection parameters.

#### Scenario: Configuration Format
- **WHEN** SVM models are configured
- **THEN** each model entry includes a name, path, and optional selection criteria
- **AND** selection criteria may include camera ID, pose confidence thresholds, or other contextual parameters