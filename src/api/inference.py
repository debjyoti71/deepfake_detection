import torch
import cv2
import librosa
import numpy as np
from pathlib import Path
from src.models.simple_models import VisualModel, AudioModel, FusionModel
from src.utils.logger import setup_logger

class DeepfakeInference:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger('DeepfakeInference')
        self.models_loaded = False
        
    def load_models(self):
        """Load trained models"""
        try:
            self.visual_model = VisualModel(num_classes=4).to(self.device)
            self.visual_model.load_state_dict(torch.load('models/visual_model.pth', map_location=self.device))
            self.visual_model.eval()
            
            self.audio_model = AudioModel(num_classes=4).to(self.device)
            self.audio_model.load_state_dict(torch.load('models/audio_model.pth', map_location=self.device))
            self.audio_model.eval()
            
            self.fusion_model = FusionModel(num_classes=4).to(self.device)
            self.fusion_model.load_state_dict(torch.load('models/fusion_model.pth', map_location=self.device))
            self.fusion_model.eval()
            
            self.models_loaded = True
            self.logger.info("All models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def extract_visual_features(self, video_path):
        """Extract visual features from video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        try:
            for i in range(8):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                else:
                    break
        finally:
            cap.release()
        
        if frames:
            return np.array(frames).mean(axis=0)
        else:
            return np.zeros((224, 224, 3))
    
    def extract_audio_features(self, video_path):
        """Extract audio features from video"""
        try:
            y, sr = librosa.load(str(video_path), sr=16000, duration=5.0)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=512)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            if mel_spec_db.shape[1] < 160:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 160 - mel_spec_db.shape[1])), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :160]
                
            return mel_spec_db
        except:
            return np.zeros((64, 160))
    
    def predict(self, video_path):
        """Predict deepfake scores for uploaded video"""
        if not self.models_loaded:
            self.load_models()
            
        self.logger.info(f"Processing video: {video_path}")
        
        # Extract features
        visual_features = self.extract_visual_features(video_path)
        audio_features = self.extract_audio_features(video_path)
        
        # Prepare tensors
        visual_tensor = torch.FloatTensor(visual_features).permute(2, 0, 1).unsqueeze(0).to(self.device)
        audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Visual prediction
            visual_output = self.visual_model(visual_tensor)
            visual_probs = torch.softmax(visual_output, dim=1)[0]
            
            # Audio prediction
            audio_output = self.audio_model(audio_tensor)
            audio_probs = torch.softmax(audio_output, dim=1)[0]
            
            # Fusion prediction
            fusion_output = self.fusion_model(visual_tensor, audio_tensor)
            fusion_probs = torch.softmax(fusion_output, dim=1)[0]
        
        # Convert to scores
        categories = ['A', 'B', 'C', 'D']
        
        results = {
            'visual': {
                'prediction': categories[visual_probs.argmax().item()],
                'confidence': visual_probs.max().item(),
                'scores': {cat: prob.item() for cat, prob in zip(categories, visual_probs)}
            },
            'audio': {
                'prediction': categories[audio_probs.argmax().item()],
                'confidence': audio_probs.max().item(),
                'scores': {cat: prob.item() for cat, prob in zip(categories, audio_probs)}
            },
            'fusion': {
                'prediction': categories[fusion_probs.argmax().item()],
                'confidence': fusion_probs.max().item(),
                'scores': {cat: prob.item() for cat, prob in zip(categories, fusion_probs)}
            }
        }
        
        self.logger.info(f"Prediction completed: {results['fusion']['prediction']} ({results['fusion']['confidence']:.3f})")
        return results