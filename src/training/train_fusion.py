import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# ----------------------------
# Paths
# ----------------------------
METADATA = "data/raw/meta_data.csv"
MODEL_OUT = "models/fusion_model/fusion_model.pth"

VISUAL_MODEL_PATH = "models/visual_model.pth"
AUDIO_MODEL_PATH = "models/audio_model.pth"
SYNC_MODEL_PATH  = "models/sync_model.pth"

BATCH_SIZE = 16
EPOCHS = 2
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset
# ----------------------------
class FusionDataset(Dataset):
    """
    Loads outputs of unimodal models for each video
    (or can use extracted features directly)
    """
    def __init__(self, metadata_csv):
        self.df = pd.read_csv(metadata_csv)
        self.samples = []

        for idx, row in self.df.iterrows():
            video_id = row['filename'].split('.')[0]
            label = 0 if row['category']=='A' else 1
            # For simplicity, assume features/predictions saved as npy
            visual_feat_path = f"data/processed/fusion_features/visual_{video_id}.npy"
            audio_feat_path  = f"data/processed/fusion_features/audio_{video_id}.npy"
            sync_feat_path   = f"data/processed/fusion_features/sync_{video_id}.npy"
            if os.path.exists(visual_feat_path) and os.path.exists(audio_feat_path) and os.path.exists(sync_feat_path):
                self.samples.append((visual_feat_path, audio_feat_path, sync_feat_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vf, af, sf, label = self.samples[idx]
        visual = torch.tensor(np.load(vf), dtype=torch.float32)
        audio  = torch.tensor(np.load(af), dtype=torch.float32)
        sync   = torch.tensor(np.load(sf), dtype=torch.float32)
        fusion_feat = torch.cat([visual, audio, sync], dim=0)
        return fusion_feat, label

# ----------------------------
# Dataloader
# ----------------------------
dataset = FusionDataset(METADATA)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ----------------------------
# Fusion Model
# ----------------------------
class FusionNet(nn.Module):
    def __init__(self, input_dim):
        super(FusionNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(x)

# Determine input dim dynamically
sample_feat, _ = dataset[0]
input_dim = sample_feat.shape[0]

model = FusionNet(input_dim).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for feats, labels in tqdm(dataloader):
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * feats.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataset)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

# ----------------------------
# Save Model
# ----------------------------
torch.save(model.state_dict(), MODEL_OUT)
print(f"✅ Saved fusion model at {MODEL_OUT}")
