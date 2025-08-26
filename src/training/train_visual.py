import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import glob
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Paths
# ----------------------------
FACES_DIR = "data/processed/faces"
METADATA = "data/metadata.csv"
MODEL_OUT = "models/visual_model.pth"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset
# ----------------------------
class VisualDataset(Dataset):
    def __init__(self, metadata_csv, faces_dir, transform=None):
        self.df = pd.read_csv(metadata_csv)
        self.faces_dir = faces_dir
        self.transform = transform

        # Only visual categories
        self.samples = []
        for idx, row in self.df.iterrows():
            video_id = os.path.splitext(row['filename'])[0]
            label = 0 if row['category'] == 'A' else 1  # 0: Real, 1: Fake
            frame_files = glob.glob(os.path.join(faces_dir, video_id, "*.jpg"))
            for f in frame_files:
                self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

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
dataset = VisualDataset(METADATA, FACES_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ----------------------------
# Model
# ----------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Real / Fake
model = model.to(DEVICE)

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

    for imgs, labels in tqdm(dataloader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
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
print(f"✅ Saved visual model at {MODEL_OUT}")
