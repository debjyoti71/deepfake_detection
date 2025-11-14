import cv2
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import os

class FeatureExtractor:
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.videos_dir = self.processed_dir / "videos"
        
    def extract_all_features(self):
        """Extract features from balanced dataset"""
        print("Extracting features from balanced dataset...")
        
        # Load balanced sample metadata
        balanced_df = pd.read_csv(self.processed_dir / "balanced_sample.csv")
        
        features = []
        for _, row in balanced_df.iterrows():
            try:
                # Get video path
                video_path = self.videos_dir / row['category'] / row['race'] / row['gender'] / row['path']
                
                if video_path.exists():
                    # Extract visual features
                    visual_features = self._extract_visual_features(video_path)
                    
                    # Extract audio features  
                    audio_features = self._extract_audio_features(video_path)
                    
                    features.append({
                        'source_id': row['source'],
                        'category': row['category'],
                        'visual_features': visual_features,
                        'audio_features': audio_features,
                        'label': self._get_label(row['category'])
                    })
                    
            except Exception as e:
                print(f"Error processing {row['path']}: {e}")
                
        print(f"Extracted features from {len(features)} videos")
        return features
    
    def _extract_visual_features(self, video_path):
        """Extract visual features from video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Extract first 8 frames
        for i in range(8):
            ret, frame = cap.read()
            if ret:
                # Resize to 224x224 and normalize
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                break
                
        cap.release()
        
        if frames:
            return np.array(frames).mean(axis=0)  # Average frame
        else:
            return np.zeros((224, 224, 3))
    
    def _extract_audio_features(self, video_path):
        """Extract audio features from video"""
        try:
            # Extract audio using librosa
            y, sr = librosa.load(str(video_path), sr=16000, duration=5.0)
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=512)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to fixed shape
            if mel_spec_db.shape[1] < 160:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 160 - mel_spec_db.shape[1])), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :160]
                
            return mel_spec_db
            
        except Exception as e:
            print(f"Audio extraction error: {e}")
            return np.zeros((64, 160))
    
    def _get_label(self, category):
        """Convert category to numeric label"""
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        return label_map.get(category, 0)