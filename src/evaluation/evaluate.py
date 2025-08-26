import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pickle

# ----------------------------
# Paths
# ----------------------------
METADATA = "data/metadata.csv"
FACES_DIR = "data/processed/faces"
AUDIO_FEAT_DIR = "data/processed/audio_features"
SYNC_DIR = "data/processed/sync"
FUSION_FEAT_DIR = "data/processed/fusion_features"
PREDICTIONS_DIR = "results/predictions"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

VISUAL_MODEL_PATH = "models/visual_model.pth"
AUDIO_MODEL_PATH  = "models/audio_model.pth"
SYNC_MODEL_PATH   = "models/sync_model.pth"
FUSION_MODEL_PATH = "models/fusion_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# ----------------------------
# Transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----------------------------
# Dataset Definitions
# ----------------------------
class VisualDataset(Dataset):
    def __init__(self, metadata_csv, faces_dir, transform=None):
        self.df = pd.read_csv(metadata_csv)
        self.faces_dir = faces_dir
        self.transform = transform
        self.samples = []
        for idx, row in self.df.iterrows():
            if row['split'] != 'test':
                continue
            video_id = row['filename'].split('.')[0]
            label = 0 if row['category']=='A' else 1
            frame_files = [f for f in os.listdir(os.path.join(faces_dir, video_id)) if f.endswith(".jpg")]
            self.samples.append((video_id, frame_files, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, frame_files, label = self.samples[idx]
        imgs = []
        for f in frame_files:
            img = Image.open(os.path.join(FACES_DIR, video_id, f)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        return video_id, imgs, label

class AudioDataset(Dataset):
    def __init__(self, metadata_csv, feat_dir):
        self.df = pd.read_csv(metadata_csv)
        self.feat_dir = feat_dir
        self.samples = []
        for idx, row in self.df.iterrows():
            if row['split'] != 'test':
                continue
            video_id = row['filename'].split('.')[0]
            label = 0 if row['category']=='A' else 1
            feat_path = os.path.join(feat_dir, f"{video_id}.npy")
            if os.path.exists(feat_path):
                self.samples.append((video_id, feat_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, feat_path, label = self.samples[idx]
        mel = torch.tensor(np.load(feat_path), dtype=torch.float32).unsqueeze(0)  # 1 x n_mels x T
        return video_id, mel, label

class SyncDataset(Dataset):
    def __init__(self, sync_dir):
        self.files = [os.path.join(sync_dir, f) for f in os.listdir(sync_dir) if f.endswith(".pkl")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with open(path, "rb") as f:
            data = pickle.load(f)
        frames = []
        for fpath in data['frames']:
            img = Image.open(fpath).convert("RGB")
            img = transform(img)
            frames.append(img)
        frames = torch.stack(frames)
        audio = torch.tensor(data['audio_features'], dtype=torch.float32).unsqueeze(0)
        video_id = os.path.basename(path).replace(".pkl","")
        label = 1  # synced; can modify for test
        return video_id, frames, audio, label

class FusionDataset(Dataset):
    def __init__(self, metadata_csv, fusion_feat_dir):
        self.df = pd.read_csv(metadata_csv)
        self.samples = []
        for idx, row in self.df.iterrows():
            if row['split'] != 'test':
                continue
            video_id = row['filename'].split('.')[0]
            label = 0 if row['category']=='A' else 1
            vf = os.path.join(fusion_feat_dir, f"visual_{video_id}.npy")
            af = os.path.join(fusion_feat_dir, f"audio_{video_id}.npy")
            sf = os.path.join(fusion_feat_dir, f"sync_{video_id}.npy")
            if os.path.exists(vf) and os.path.exists(af) and os.path.exists(sf):
                self.samples.append((video_id, vf, af, sf, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, vf, af, sf, label = self.samples[idx]
        visual = torch.tensor(np.load(vf), dtype=torch.float32)
        audio  = torch.tensor(np.load(af), dtype=torch.float32)
        sync   = torch.tensor(np.load(sf), dtype=torch.float32)
        fusion_feat = torch.cat([visual,audio,sync], dim=0)
        return video_id, fusion_feat, label

# ----------------------------
# Model Loading
# ----------------------------
def load_visual_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(VISUAL_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_audio_model():
    # Simple CNN from previous step
    class AudioCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1,16,3,padding=1)
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16,32,3,padding=1)
            self.pool2 = nn.MaxPool2d(2)
            self.fc = nn.Linear(32*40*40,2)  # adjust 40*40 if needed
        def forward(self,x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool2(torch.relu(self.conv2(x)))
            x = x.view(x.size(0),-1)
            return self.fc(x)
    model = AudioCNN()
    model.load_state_dict(torch.load(AUDIO_MODEL_PATH,map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_sync_model():
    # Simple SyncNet placeholder
    class SyncNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64,2)  # placeholder
        def forward(self,x1,x2):
            x = torch.cat([x1.mean(dim=0), x2.mean(dim=1)],dim=-1).unsqueeze(0)
            return self.fc(x)
    model = SyncNet()
    model.load_state_dict(torch.load(SYNC_MODEL_PATH,map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_fusion_model(input_dim):
    class FusionNet(nn.Module):
        def __init__(self,input_dim):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim,128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128,2)
            )
        def forward(self,x):
            return self.fc(x)
    model = FusionNet(input_dim)
    model.load_state_dict(torch.load(FUSION_MODEL_PATH,map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate_model(model, dataloader, model_type='visual'):
    all_preds = []
    all_labels = []
    video_ids = []

    with torch.no_grad():
        for batch in dataloader:
            if model_type=='visual':
                vid, imgs, labels = batch
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                probs = torch.softmax(outputs,dim=1)[:,1].mean().item()
                pred = 1 if probs>=0.5 else 0
            elif model_type=='audio':
                vid, audio, labels = batch
                audio = audio.to(DEVICE)
                outputs = model(audio)
                probs = torch.softmax(outputs,dim=1)[:,1].mean().item()
                pred = 1 if probs>=0.5 else 0
            elif model_type=='fusion':
                vid, feats, labels = batch
                feats = feats.to(DEVICE)
                outputs = model(feats)
                probs = torch.softmax(outputs,dim=1)[:,1].item()
                pred = 1 if probs>=0.5 else 0
            # Add sync evaluation if needed

            video_ids.append(vid[0] if isinstance(vid,list) else vid)
            all_preds.append(pred)
            all_labels.append(labels if isinstance(labels,int) else labels[0])

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    print(f"{model_type.capitalize()} Model - Acc: {acc:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print(report)

    # Save predictions
    df = pd.DataFrame({'video_id': video_ids, 'pred': all_preds, 'true': all_labels})
    df.to_csv(os.path.join(PREDICTIONS_DIR,f'{model_type}_predictions.csv'),index=False)

# ----------------------------
# Main Evaluation
# ----------------------------
if __name__=="__main__":
    print("Evaluating Visual Model...")
    visual_model = load_visual_model()
    visual_loader = DataLoader(VisualDataset(METADATA,FACES_DIR,transform),batch_size=1)
    evaluate_model(visual_model, visual_loader,'visual')

    print("Evaluating Audio Model...")
    audio_model = load_audio_model()
    audio_loader = DataLoader(AudioDataset(METADATA,AUDIO_FEAT_DIR),batch_size=1)
    evaluate_model(audio_model, audio_loader,'audio')

    print("Evaluating Fusion Model...")
    # placeholder input_dim; adjust based on your feature concat
    sample_feat = np.load(os.path.join(FUSION_FEAT_DIR,os.listdir(FUSION_FEAT_DIR)[0]))
    fusion_model = load_fusion_model(sample_feat.shape[0])
    fusion_loader = DataLoader(FusionDataset(METADATA,FUSION_FEAT_DIR),batch_size=1)
    evaluate_model(fusion_model,fusion_loader,'fusion')
