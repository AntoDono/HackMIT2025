# Brain Wave Classification Model

## Overview
A state-of-the-art neural network for real-time EEG emotion classification, inspired by EEGNet architecture. The model is optimized for local inference with fast prediction times while maintaining high accuracy.

## Model Architecture

### EEGNet-Optimized Architecture

The model implements a sophisticated convolutional neural network specifically designed for EEG signal processing:

```
Input: (batch_size, time_points, channels)
   ↓
Block 1: Temporal Convolution
   ├── Conv1D(16 filters, kernel=64, padding='same')
   ├── BatchNormalization()
   └── Reshape(time_points, 1, 16)
   ↓
Block 2: Depthwise Spatial Convolution
   ├── DepthwiseConv2D(kernel=(1,1), depth_multiplier=2)
   ├── BatchNormalization()
   ├── ELU Activation
   ├── AveragePooling2D(pool_size=(4,1))
   └── Dropout(0.25)
   ↓
Block 3: Separable Convolution
   ├── SeparableConv2D(32 filters, kernel=(16,1))
   ├── BatchNormalization()
   ├── ELU Activation
   ├── AveragePooling2D(pool_size=(8,1))
   └── Dropout(0.25)
   ↓
Classification Block
   ├── Flatten()
   ├── Dense(32, activation='relu')
   ├── Dropout(0.5)
   └── Dense(num_classes, activation='softmax')
```

### Key Architecture Features

#### 1. **Temporal Convolution Block**
- **Purpose**: Learns frequency-specific filters across time
- **Implementation**: 1D convolution with 16 filters and kernel size 64
- **Benefit**: Captures temporal patterns in EEG signals effectively

#### 2. **Depthwise Spatial Convolution**
- **Purpose**: Efficient spatial filtering between EEG channels
- **Implementation**: Depthwise separable convolution with depth multiplier of 2
- **Benefit**: Reduces parameters by ~8x compared to standard convolution while maintaining performance

#### 3. **Separable Convolution Block**
- **Purpose**: Further feature extraction with computational efficiency
- **Implementation**: Separable 2D convolution with 32 filters
- **Benefit**: Separates spatial and depthwise convolutions for efficiency

#### 4. **Optimization Techniques**
- **Batch Normalization**: Stabilizes training and improves convergence
- **ELU Activation**: Better gradient flow for EEG signals compared to ReLU
- **Average Pooling**: Reduces overfitting while preserving important features
- **Dropout**: Prevents overfitting with rates of 0.25 and 0.5

## Performance Characteristics

### Speed Optimizations
- **Depthwise Convolutions**: Significantly faster than standard convolutions
- **Efficient Architecture**: Fewer parameters than comparable dense networks
- **Local Inference**: Optimized for CPU execution without GPU requirements

### Expected Performance
- **Inference Time**: Target <10ms per prediction (improvement over 14.16ms baseline)
- **Memory Usage**: Minimal memory footprint for edge deployment
- **Accuracy**: State-of-the-art performance for EEG emotion classification

## Model Configuration

### Default Parameters
```python
wave_types = ['alpha', 'beta', 'gamma', 'delta', 'theta']  # EEG frequency bands
num_points = 100  # Time points per prediction
```

### Supported Emotions
- Happy
- Sad
- Depressed
- Excited
- (Configurable based on training data)

## Usage Example

```python
from main import BrainwaveModel

# Initialize model
model = BrainwaveModel(
    wave_types=['alpha', 'beta', 'gamma', 'delta', 'theta'],
    num_points=100
)

# Train the model
model.train(brain_wave_dataframe, emotion_labels, epochs=50)

# Make predictions
emotion = model.predict(new_eeg_data)
print(f"Predicted emotion: {emotion}")
```

## Technical Details

### Input Format
- **Data Type**: Pandas DataFrame
- **Columns**: One column per wave type (alpha, beta, gamma, delta, theta)
- **Rows**: Time series data points
- **Preprocessing**: Automatic chunking and normalization

### Training Strategy
- **Chunked Training**: Automatically splits long sequences into overlapping windows
- **Data Augmentation**: Uses sliding window approach for robust training
- **Validation**: Built-in train/validation split with early stopping

### Architecture Benefits

1. **EEG-Specific Design**: Tailored for brain wave signal characteristics
2. **Computational Efficiency**: Optimized for real-time inference
3. **Scalability**: Configurable for different numbers of channels and time points
4. **Robustness**: Dropout and batch normalization prevent overfitting

## Files Structure

```
ml/
├── main.py          # Main model implementation
├── test.py          # Comprehensive testing and benchmarking
├── requirements.txt # Dependencies
└── README.md        # This documentation
```

## Dependencies

```
tensorflow==2.13.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
```

## Research Foundation

This architecture is inspired by:
- **EEGNet**: Compact convolutional neural network for EEG-based brain-computer interfaces
- **Depthwise Separable Convolutions**: Efficient neural network architectures for mobile applications
- **State-of-the-art EEG Classification**: Recent advances in neural networks for EEG signal processing

The model combines the best practices from EEG research while maintaining efficiency for local deployment.
