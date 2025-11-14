# Balanced Sampling Implementation Summary

## Overview
Successfully implemented balanced sampling strategy that ensures for each real-real video (Category A), we get corresponding fake versions from the same source person across all categories (B, C, D).

## Key Features

### 1. Balanced Dataset Structure
- **Category A (Real-Real)**: 25 samples - Original authentic videos
- **Category B (Real-Fake)**: 25 samples - Same video with synthetic audio (RTVC voice cloning)
- **Category C (Fake-Real)**: 25 samples - Face-swapped video with original audio (Faceswap/FSGAN)
- **Category D (Fake-Fake)**: 24 samples - Both face and audio manipulated (combination methods)

### 2. Source Person Consistency
Each group of 4 videos (A, B, C, D) comes from the same source person ID, ensuring:
- **Visual consistency**: Same person's face across all categories
- **Audio consistency**: Same person's voice in categories A and C
- **Controlled comparison**: Model can learn differences between real and fake from identical source

### 3. Dataset Statistics
- **Total samples**: 99 videos
- **Race**: 100% Asian (East) - focused dataset for better learning
- **Gender distribution**: 55 men, 44 women - balanced representation
- **Missing samples**: Only 1 missing (Category D for id04789)

## Implementation Details

### Files Created
1. `src/preprocessing/balanced_sampler.py` - Core balanced sampling logic
2. `src/preprocessing/data_processor.py` - Updated data processor using balanced sampling
3. `data/processed/balanced_sample.csv` - Generated balanced dataset
4. Updated `.env` configuration for balanced sampling

### Configuration
```env
MAX_SAMPLES_PER_REAL=25
USE_BALANCED_SAMPLING=True
FILTER_BY_RACE=True
SELECTED_RACES=Asian (East)
```

## Benefits for Model Training

### 1. Better Learning of Differences
- Model can directly compare real vs fake versions of the same person
- Reduces bias from person-specific features
- Focuses learning on manipulation artifacts rather than identity differences

### 2. Balanced Class Distribution
- Nearly equal samples across all categories (25:25:25:24)
- Prevents model bias toward any particular category
- Ensures robust multi-class classification

### 3. Controlled Experimental Setup
- Same source person across categories eliminates confounding variables
- Enables precise evaluation of detection capabilities
- Supports better generalization to unseen identities

## Example Sample Group
For source person `id06443`:
- **A**: `00232.mp4` (Real video, real audio)
- **B**: `00232_fake.mp4` (Real video, synthetic audio via RTVC)
- **C**: `00232_id01451_VdiJbjp23Fc.mp4` (Face-swapped with id01451, real audio)
- **D**: `00232_id09171_B-JRLp6chC0_id00566_wavtolip.mp4` (Face-swapped + synthetic audio)

## Next Steps
1. **Feature Extraction**: Extract visual and audio features from balanced dataset
2. **Model Training**: Train individual models (visual, audio, sync) on balanced data
3. **Fusion Training**: Train fusion model to combine predictions from balanced samples
4. **Evaluation**: Test model performance on balanced test set

This balanced sampling approach ensures the deepfake detection model learns to distinguish between authentic and manipulated content based on actual manipulation artifacts rather than identity-specific differences, leading to more robust and generalizable detection capabilities.