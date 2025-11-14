# Model Architectures Used

## Current Simple Models

### 1. Visual Model (CNN)
- **Input**: 224×224×3 RGB frames
- **Architecture**: 
  - Conv2D(3→32) + ReLU + MaxPool2D
  - Conv2D(32→64) + ReLU + MaxPool2D
  - Flatten + FC(64×56×56→512) + Dropout(0.5)
  - FC(512→4) for classification
- **Parameters**: ~200K
- **Purpose**: Face manipulation detection

### 2. Audio Model (CNN)
- **Input**: 64×160 Mel-spectrograms
- **Architecture**:
  - Conv2D(1→32) + ReLU + MaxPool2D
  - Conv2D(32→64) + ReLU + MaxPool2D
  - Flatten + FC(64×16×40→256) + Dropout(0.5)
  - FC(256→4) for classification
- **Parameters**: ~100K
- **Purpose**: Voice synthesis detection

### 3. Fusion Model (Multi-modal)
- **Input**: Visual + Audio features
- **Architecture**:
  - Visual branch: VisualModel → 256 features
  - Audio branch: AudioModel → 256 features
  - Concatenate: [256+256] = 512 features
  - FC(512→4) + Dropout(0.5)
- **Parameters**: ~300K
- **Purpose**: Combined decision making

## Potential Advanced Models

### Visual: ResNet-18/EfficientNet
### Audio: Transformer/LSTM
### Fusion: Attention-based fusion

## Current Performance
- Visual: 25% accuracy
- Audio: 55% accuracy  
- Fusion: 55% accuracy (best)