import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Paths
# ----------------------------
FEATURE_DIR = "data/processed/audio_features"
METADATA = "data/metadata.csv"
MODEL_OUT = "models/audio_model.pth"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset
# ----------------------------
class AudioDataset(Dataset):
    def __init__(self, metadata_csv, feature_dir):
        self.df = pd.read_csv(metadata_csv)
        self.feature_dir = feature_dir
        self.samples = []

        for idx, row in self.df.iterrows():
            video_id = os.path.splitext(row['filename'])[0]
            label = 0 if row['category'] == 'A' else 1  # 0: Real, 1: Fake
            feature_path = os.path.join(feature_dir, f"{video_id}.npy")
            if os.path.exists(feature_path):
                self.samples.append((feature_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, label = self.samples[idx]
        mel = np.load(feat_path)  # shape: n_mels x T
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # 1 x n_mels x T
        return mel, label

# ----------------------------
# Dataloader
# ----------------------------
dataset = AudioDataset(METADATA, FEATURE_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ----------------------------
# Model (Simple CNN)
# ----------------------------
class AudioCNN(nn.Module):
    def __init__(self, n_classes=2):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(16,32,kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*40*40, 128)  # adjust 40*40 if input shape differs
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = AudioCNN().to(DEVICE)
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

    for mel, labels in tqdm(dataloader):
        mel, labels = mel.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(mel)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mel.size(0)
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
print(f"✅ Saved audio model at {MODEL_OUT}")
