import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Paths and hyperparameters
SYNC_DIR = "data/processed/sync"
MODEL_OUT = "models/sync_model/sync_model.pth"
BATCH_SIZE = 8
EPOCHS = 2
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {DEVICE}")

# Transform applied on each frame image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset to load synchronized frame + audio feature data
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

        # Load frames: list of file paths
        frames = []
        for fpath in data['frames']:
            img = Image.open(fpath).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        frames = torch.stack(frames)  # [T, 3, H, W]

        # Load audio features: shape [n_mels, T]
        audio = torch.tensor(data['audio_features'], dtype=torch.float32)
        audio = audio.unsqueeze(0)  # [1, n_mels, T]

        # Label: 1 means synced (for now, only positive samples)
        label = torch.tensor(1, dtype=torch.long)

        return frames, audio, label

def pad_collate(batch):
    frames_batch, audio_batch, labels = zip(*batch)

    # Pad frames
    max_frames_len = max([f.size(0) for f in frames_batch])
    frames_padded = []
    for frames in frames_batch:
        T, C, H, W = frames.shape
        if T < max_frames_len:
            pad = torch.zeros((max_frames_len - T, C, H, W), dtype=frames.dtype)
            frames = torch.cat([frames, pad], dim=0)
        frames_padded.append(frames)
    frames_padded = torch.stack(frames_padded)  # [B, max_frames_len, 3, H, W]

    # Pad audio (along temporal dim, last dim)
    max_audio_len = max([a.size(-1) for a in audio_batch])
    audio_padded = []
    for audio in audio_batch:
        B, n_mels, T_audio = audio.shape  # B=1 usually
        if T_audio < max_audio_len:
            pad = torch.zeros((B, n_mels, max_audio_len - T_audio), dtype=audio.dtype)
            audio = torch.cat([audio, pad], dim=2)
        audio_padded.append(audio)
    audio_padded = torch.stack(audio_padded)  # [B, 1, n_mels, max_audio_len]

    labels = torch.stack(labels)

    return frames_padded, audio_padded, labels


# Model definition
class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()
        # Visual encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Audio encoder
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Classifier
        self.fc = nn.Linear(32 + 32, 2)  # binary classification

    def forward(self, frames, audio):
        """
        Args:
            frames: [B, T, 3, H, W]
            audio: [B, 1, n_mels, T]

        Returns:
            logits: [B, 2]
        """
        # Average frames over time dimension
        frames_mean = torch.mean(frames, dim=1)  # [B, 3, H, W]
        f_feat = self.cnn(frames_mean)
        f_feat = f_feat.view(f_feat.size(0), -1)  # [B, 32]

        a_feat = self.audio_cnn(audio)
        a_feat = a_feat.view(a_feat.size(0), -1)  # [B, 32]

        x = torch.cat([f_feat, a_feat], dim=1)  # [B, 64]
        out = self.fc(x)
        return out

def train():
    dataset = SyncDataset(SYNC_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, collate_fn=pad_collate)

    model = SyncNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for frames, audio, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            frames, audio, labels = frames.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(frames, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * frames.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"✅ Saved sync model at {MODEL_OUT}")

if __name__ == "__main__":
    train()
