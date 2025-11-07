## 1. Design Model Selection Logic
- [ ] 1.1 Define model selection criteria (e.g., by camera ID, pose type, confidence threshold)
- [ ] 1.2 Design configuration format for multiple models
- [ ] 1.3 Plan backward compatibility with single model usage

## 2. Update Configuration
- [ ] 2.1 Modify Config class to support multiple SVM model paths
- [ ] 2.2 Add model selection parameters to command-line arguments
- [ ] 2.3 Update configuration validation

## 3. Modify PoseProcessor Class
- [ ] 3.1 Update __init__ to accept multiple SVM models
- [ ] 3.2 Implement model selection logic in anomaly detection
- [ ] 3.3 Add fallback to default model when no specific model matches

## 4. Update Main Function
- [ ] 4.1 Modify main() to load multiple SVM models based on configuration
- [ ] 4.2 Update PoseProcessor instantiation with multiple models
- [ ] 4.3 Add error handling for missing models

## 5. Testing and Validation
- [ ] 5.1 Test backward compatibility with single model
- [ ] 5.2 Test multiple model loading and selection
- [ ] 5.3 Validate anomaly detection with different models