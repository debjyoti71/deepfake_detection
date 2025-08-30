import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ----------------------------
# Paths
# ----------------------------
SYNC_DIR = "data/processed/sync"
MODEL_OUT = "models/sync_model/sync_model.pth"
BATCH_SIZE = 8
EPOCHS = 2
LR = 1e-3
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
# Custom collate_fn to pad variable length frames
# ----------------------------
def pad_collate(batch):
    """
    batch: list of tuples (frames, audio, label)
    frames: [T, 3, H, W], variable T per sample
    """
    max_frames = max(item[0].size(0) for item in batch)

    frames_padded = []
    audios = []
    labels = []

    for frames, audio, label in batch:
        T = frames.size(0)
        if T < max_frames:
            pad_tensor = torch.zeros((max_frames - T, *frames.shape[1:]), dtype=frames.dtype)
            frames = torch.cat([frames, pad_tensor], dim=0)
        frames_padded.append(frames)
        audios.append(audio)
        labels.append(label)

    frames_batch = torch.stack(frames_padded, dim=0)  # [B, T_max, 3, 224, 224]
    audios_batch = torch.stack(audios, dim=0)         # [B, 1, n_mels, T_audio]
    labels_batch = torch.tensor(labels, dtype=torch.long)

    return frames_batch, audios_batch, labels_batch

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
        """
        frames: [B, T, 3, H, W]
        audio:  [B, 1, n_mels, T_audio]
        """
        # Mean over temporal dimension (frames)
        frames = torch.mean(frames, dim=1)  # [B, 3, H, W]
        f_feat = self.cnn(frames)            # [B, 32, 1, 1]
        f_feat = f_feat.view(f_feat.size(0), -1)  # [B, 32]

        a_feat = self.audio_cnn(audio)      # [B, 32, 1, 1]
        a_feat = a_feat.view(a_feat.size(0), -1)  # [B, 32]

        x = torch.cat([f_feat, a_feat], dim=1)    # [B, 64]
        out = self.fc(x)                          # [B, 2]
        return out

if __name__ == '__main__':
    # ----------------------------
    # Dataloader
    # ----------------------------
    dataset = SyncDataset(SYNC_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=pad_collate)

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
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    # ----------------------------
    # Save Model
    # ----------------------------
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"✅ Saved sync model at {MODEL_OUT}")
