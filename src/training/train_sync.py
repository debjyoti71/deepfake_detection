import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# ----------------------------
# Paths
# ----------------------------
SYNC_DIR = "data/processed/sync"
MODEL_OUT = "models/sync_model.pth"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset
# ----------------------------
class SyncDataset(Dataset):
    def __init__(self, sync_dir, transform=None):
        self.files = [os.path.join(sync_dir, f) for f in os.listdir(sync_dir) if f.endswith(".pkl")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Load frames
        frames = []
        for fpath in data['frames']:
            img = Image.open(fpath).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        frames = torch.stack(frames)  # shape: [T, 3, H, W]

        # Load audio features
        audio = torch.tensor(data['audio_features'], dtype=torch.float32)  # shape: [n_mels, T]
        audio = audio.unsqueeze(0)  # [1, n_mels, T]

        # Sync label: real pair = 1, mismatched = 0
        label = 1  # assuming synced data; you can create negative pairs on-the-fly in collate_fn

        return frames, audio, label

# ----------------------------
# Transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----------------------------
# Dataloader
# ----------------------------
dataset = SyncDataset(SYNC_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ----------------------------
# Model
# ----------------------------
class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()
        # Visual encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Audio encoder
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Classifier
        self.fc = nn.Linear(32+32, 2)  # 2 classes: synced / mismatched

    def forward(self, frames, audio):
        # Take mean over temporal dimension
        frames = torch.mean(frames, dim=0)  # [3,H,W]
        frames = frames.unsqueeze(0)  # add batch dim
        f_feat = self.cnn(frames)
        f_feat = f_feat.view(f_feat.size(0), -1)

        a_feat = self.audio_cnn(audio)
        a_feat = a_feat.view(a_feat.size(0), -1)

        x = torch.cat([f_feat, a_feat], dim=1)
        out = self.fc(x)
        return out

model = SyncNet().to(DEVICE)
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

    for frames, audio, labels in tqdm(dataloader):
        frames, audio, labels = frames.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(frames, audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * frames.size(0)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    epoch_loss = running_loss / len(dataset)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

# ----------------------------
# Save Model
# ----------------------------
torch.save(model.state_dict(), MODEL_OUT)
print(f"✅ Saved sync model at {MODEL_OUT}")
