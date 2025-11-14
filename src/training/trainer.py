import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from src.models.simple_models import VisualModel, AudioModel, FusionModel
from src.preprocessing.feature_extractor import FeatureExtractor
from src.utils.logger import setup_logger

class DeepfakeDataset(Dataset):
    def __init__(self, features, mode='fusion'):
        self.features = features
        self.mode = mode
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = self.features[idx]
        
        if self.mode == 'visual':
            return torch.FloatTensor(item['visual_features']).permute(2, 0, 1), torch.LongTensor([item['label']])
        elif self.mode == 'audio':
            return torch.FloatTensor(item['audio_features']).unsqueeze(0), torch.LongTensor([item['label']])
        else:  # fusion
            visual = torch.FloatTensor(item['visual_features']).permute(2, 0, 1)
            audio = torch.FloatTensor(item['audio_features']).unsqueeze(0)
            return {'visual': visual, 'audio': audio, 'label': torch.LongTensor([item['label']])}

class ModelTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.logger = setup_logger('ModelTrainer')
        
    def train_all_models(self):
        """Train all models on balanced dataset"""
        self.logger.info("Starting model training on balanced dataset")
        
        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_all_features()
        
        if not features:
            self.logger.warning("No features extracted. Skipping training.")
            return
            
        # Split data (simple 80-20 split)
        split_idx = int(0.8 * len(features))
        train_features = features[:split_idx]
        val_features = features[split_idx:]
        
        self.logger.info(f"Training on {len(train_features)} samples, validating on {len(val_features)} samples")
        
        # Train individual models
        self.train_visual_model(train_features, val_features)
        self.train_audio_model(train_features, val_features)
        self.train_fusion_model(train_features, val_features)
        
    def train_visual_model(self, train_features, val_features):
        """Train visual model"""
        self.logger.info("Starting visual model training")
        
        model = VisualModel(num_classes=4).to(self.device)
        train_dataset = DeepfakeDataset(train_features, mode='visual')
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(20):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.squeeze().to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            self.logger.info(f"Visual Epoch {epoch+1}/20, Loss: {total_loss/len(train_loader):.4f}")
        
        torch.save(model.state_dict(), self.models_dir / "visual_model.pth")
        self.logger.info("Visual model saved successfully")
        
    def train_audio_model(self, train_features, val_features):
        """Train audio model"""
        self.logger.info("Starting audio model training")
        
        model = AudioModel(num_classes=4).to(self.device)
        train_dataset = DeepfakeDataset(train_features, mode='audio')
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(20):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.squeeze().to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            self.logger.info(f"Audio Epoch {epoch+1}/20, Loss: {total_loss/len(train_loader):.4f}")
        
        torch.save(model.state_dict(), self.models_dir / "audio_model.pth")
        self.logger.info("Audio model saved successfully")
        
    def train_fusion_model(self, train_features, val_features):
        """Train fusion model"""
        self.logger.info("Starting fusion model training")
        
        model = FusionModel(num_classes=4).to(self.device)
        train_dataset = DeepfakeDataset(train_features, mode='fusion')
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(20):
            total_loss = 0
            for batch_idx, batch_data in enumerate(train_loader):
                visual_data = batch_data['visual'].to(self.device)
                audio_data = batch_data['audio'].to(self.device)
                target = batch_data['label'].squeeze().to(self.device)
                
                optimizer.zero_grad()
                output = model(visual_data, audio_data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            self.logger.info(f"Fusion Epoch {epoch+1}/20, Loss: {total_loss/len(train_loader):.4f}")
        
        torch.save(model.state_dict(), self.models_dir / "fusion_model.pth")
        self.logger.info("Fusion model saved successfully")