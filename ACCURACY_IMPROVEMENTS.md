# Improving Model Accuracy

## Current Results (Proof of Concept)
- Visual: 25% (random = 25%)
- Audio: 61% (better than random)
- Fusion: 69% (best performance)

## Why Low Accuracy?

### 1. Minimal Models
- Only 2 conv layers each
- No transfer learning
- No advanced architectures

### 2. Insufficient Training
- Only 5 epochs (need 50-100+)
- Small batch sizes
- No data augmentation

### 3. Basic Features
- Visual: Average frames (need face detection)
- Audio: Simple spectrograms (need advanced features)

## Immediate Improvements

### 1. Increase Training Epochs
```python
# In trainer.py, change:
for epoch in range(50):  # Instead of 5
```

### 2. Better Models
```python
# Use ResNet for visual:
from torchvision.models import resnet18
model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 4)
```

### 3. Data Augmentation
```python
# Add transforms:
transforms.RandomHorizontalFlip()
transforms.RandomRotation(10)
transforms.ColorJitter()
```

## Expected Improvements
- Visual: 25% → 70-80%
- Audio: 61% → 80-85%
- Fusion: 69% → 85-90%

## Advanced Improvements
1. Face detection + cropping
2. Temporal modeling (LSTM/Transformer)
3. Attention mechanisms
4. Ensemble methods
5. Advanced loss functions