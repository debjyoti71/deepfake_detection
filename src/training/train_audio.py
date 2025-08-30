import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Paths & Hyperparameters
# ----------------------------
FEATURE_DIR = "data/processed/audio_features"
METADATA = "data/raw/meta_data.csv"
MODEL_OUT = "models/audio_model/audio_model.pth"
BATCH_SIZE = 16
EPOCHS = 2
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset
# ----------------------------
class AudioDataset(Dataset):
    def __init__(self, metadata_csv, feature_dir):
        self.df = pd.read_csv(metadata_csv)
        self.feature_dir = feature_dir
        self.samples = []
        
        filtered_df = self.df[self.df['race'].isin(["Asian (East)", "Asian (South)"])]

        for idx, row in filtered_df.iterrows():
            try:
                video_id = os.path.splitext(row['path'])[0]
                label = 1 if row['category'] in ['C', 'D'] else 0
                feature_path = os.path.join(feature_dir, f"{video_id}.npy")
                if os.path.exists(feature_path):
                    self.samples.append((feature_path, label))
                else:
                    print(f"⚠️ Feature file not found: {feature_path}")
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, label = self.samples[idx]
        try:
            mel = np.load(feat_path)  # shape: (n_mels, T)
            # print(f"📥 Loaded {feat_path} with shape {mel.shape}")
            
            # Set a fixed length (T)
            target_len = 512
            if mel.shape[1] < target_len:
                # Pad
                pad_width = target_len - mel.shape[1]
                mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
            elif mel.shape[1] > target_len:
                # Truncate
                mel = mel[:, :target_len]

            mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # shape: (1, 80, 512)
            return mel, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"❌ Error loading file {feat_path}: {e}")
            dummy = torch.zeros((1, 80, 512), dtype=torch.float32)
            return dummy, torch.tensor(0, dtype=torch.long)


# ----------------------------
# Model
# ----------------------------
class AudioCNN(nn.Module):
    def __init__(self, n_classes=2):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        try:
            # print(f"🔁 Forward input shape: {x.shape}")
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            # print(f"✅ After conv layers: {x.shape}")
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        except Exception as e:
            print(f"❌ Model forward error: {e}")
            raise

# ----------------------------
# Main Training Script
# ----------------------------
if __name__ == "__main__":
    # Data
    dataset = AudioDataset(METADATA, FEATURE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Model, Loss, Optimizer
    model = AudioCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (mel, labels) in enumerate(tqdm(dataloader)):
            try:
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

                print(f"[Batch {batch_idx}] Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"❌ Training error on batch {batch_idx}: {e}")

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total
        print(f"📊 Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    # Save Model
    try:
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"✅ Saved audio model at {MODEL_OUT}")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
