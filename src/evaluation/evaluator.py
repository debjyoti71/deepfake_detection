import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from src.models.simple_models import VisualModel, AudioModel, FusionModel
from src.preprocessing.feature_extractor import FeatureExtractor
from src.training.trainer import DeepfakeDataset
from torch.utils.data import DataLoader

class ModelEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = Path("models")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("Evaluating trained models...")
        
        # Get test features
        extractor = FeatureExtractor()
        features = extractor.extract_all_features()
        
        if not features:
            print("No features available for evaluation")
            return
            
        # Use last 20% as test set
        split_idx = int(0.8 * len(features))
        test_features = features[split_idx:]
        
        print(f"Evaluating on {len(test_features)} test samples")
        
        # Evaluate each model
        self.evaluate_visual_model(test_features)
        self.evaluate_audio_model(test_features)
        self.evaluate_fusion_model(test_features)
        
    def evaluate_visual_model(self, test_features):
        """Evaluate visual model"""
        print("Evaluating visual model...")
        
        if not (self.models_dir / "visual_model.pth").exists():
            print("Visual model not found. Skipping evaluation.")
            return
            
        model = VisualModel(num_classes=4).to(self.device)
        model.load_state_dict(torch.load(self.models_dir / "visual_model.pth", map_location=self.device))
        model.eval()
        
        test_dataset = DeepfakeDataset(test_features, mode='visual')
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1).cpu().numpy()
                
                y_true.extend(target.squeeze().numpy())
                y_pred.extend(pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Visual Model Accuracy: {accuracy:.4f}")
        
        # Save confusion matrix
        self._save_confusion_matrix(y_true, y_pred, "Visual", ['A', 'B', 'C', 'D'])
        
    def evaluate_audio_model(self, test_features):
        """Evaluate audio model"""
        print("Evaluating audio model...")
        
        if not (self.models_dir / "audio_model.pth").exists():
            print("Audio model not found. Skipping evaluation.")
            return
            
        model = AudioModel(num_classes=4).to(self.device)
        model.load_state_dict(torch.load(self.models_dir / "audio_model.pth", map_location=self.device))
        model.eval()
        
        test_dataset = DeepfakeDataset(test_features, mode='audio')
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1).cpu().numpy()
                
                y_true.extend(target.squeeze().numpy())
                y_pred.extend(pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Audio Model Accuracy: {accuracy:.4f}")
        
        self._save_confusion_matrix(y_true, y_pred, "Audio", ['A', 'B', 'C', 'D'])
        
    def evaluate_fusion_model(self, test_features):
        """Evaluate fusion model"""
        print("Evaluating fusion model...")
        
        if not (self.models_dir / "fusion_model.pth").exists():
            print("Fusion model not found. Skipping evaluation.")
            return
            
        model = FusionModel(num_classes=4).to(self.device)
        model.load_state_dict(torch.load(self.models_dir / "fusion_model.pth", map_location=self.device))
        model.eval()
        
        test_dataset = DeepfakeDataset(test_features, mode='fusion')
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch_data in test_loader:
                visual_data = batch_data['visual'].to(self.device)
                audio_data = batch_data['audio'].to(self.device)
                target = batch_data['label'].squeeze()
                
                output = model(visual_data, audio_data)
                pred = output.argmax(dim=1).cpu().numpy()
                
                y_true.extend(target.numpy())
                y_pred.extend(pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Fusion Model Accuracy: {accuracy:.4f}")
        
        self._save_confusion_matrix(y_true, y_pred, "Fusion", ['A', 'B', 'C', 'D'])
        
        # Save detailed report
        report = classification_report(y_true, y_pred, target_names=['A', 'B', 'C', 'D'])
        with open(self.results_dir / "evaluation_report.txt", 'w') as f:
            f.write("Deepfake Detection Evaluation Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Fusion Model Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        print(f"Evaluation report saved to {self.results_dir / 'evaluation_report.txt'}")
        
    def _save_confusion_matrix(self, y_true, y_pred, model_name, class_names):
        """Save confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{model_name}_confusion_matrix.png', dpi=150)
        plt.close()
        
        print(f"{model_name} confusion matrix saved")