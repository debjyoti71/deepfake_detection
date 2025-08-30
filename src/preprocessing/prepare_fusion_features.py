# prepare_fusion_features.py
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from src.models.visual_model import VisualModel
from src.models.audio_model import AudioModel
from src.models.sync_model import SyncNet

# ----------------------------
# Paths
# ----------------------------
FRAMES_DIR = "data/processed/frames"
AUDIO_FEATURE_DIR = "data/processed/audio_features"
SYNC_DIR = "data/processed/sync"
OUT_DIR = "data/processed/fusion_features"
os.makedirs(OUT_DIR, exist_ok=True)

VISUAL_MODEL_PATH = "models/visual_model/visual_model.pth"
AUDIO_MODEL_PATH = "models/audio_model/audio_model.pth"
SYNC_MODEL_PATH  = "models/sync_model/sync_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----------------------------
# Load Models
# ----------------------------
visual_model = VisualModel()
checkpoint = torch.load(VISUAL_MODEL_PATH, map_location=DEVICE)
state_dict = {k: v for k, v in checkpoint.items() if not k.startswith("fc")}
visual_model.load_state_dict(state_dict, strict=False)
visual_model.to(DEVICE)


audio_model = AudioModel().to(DEVICE)
audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=DEVICE))
audio_model.eval()

sync_model = SyncNet().to(DEVICE)
sync_model.load_state_dict(torch.load(SYNC_MODEL_PATH, map_location=DEVICE))
sync_model.eval()

# ----------------------------
# 1. Extract visual embeddings
# ----------------------------
print("Extracting visual embeddings...")
for video_id in os.listdir(FRAMES_DIR):
    frame_folder = os.path.join(FRAMES_DIR, video_id)
    if not os.path.isdir(frame_folder):
        continue
    frame_paths = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".jpg")])
    if len(frame_paths) == 0:
        continue
    frames = torch.stack([transform(Image.open(p).convert("RGB")) for p in frame_paths]).to(DEVICE)
    with torch.no_grad():
        v_feat = visual_model(frames)  # [embedding_dim]
        v_feat = v_feat.cpu().numpy()
    np.save(os.path.join(OUT_DIR, f"visual_{video_id}.npy"), v_feat)

# ----------------------------
# 2. Extract audio embeddings
# ----------------------------
print("Extracting audio embeddings...")
for fname in os.listdir(AUDIO_FEATURE_DIR):
    if not fname.endswith(".npy"): 
        continue
    video_id = os.path.splitext(fname)[0]
    audio_feat = np.load(os.path.join(AUDIO_FEATURE_DIR, fname))
    audio_feat = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, n_mels, T]
    with torch.no_grad():
        a_feat = audio_model(audio_feat)
        a_feat = a_feat.cpu().numpy()
    np.save(os.path.join(OUT_DIR, f"audio_{video_id}.npy"), a_feat)

# ----------------------------
# 3. Extract sync embeddings
# ----------------------------
print("Extracting sync embeddings...")
for fname in os.listdir(SYNC_DIR):
    if not fname.endswith(".pkl"):
        continue
    video_id = os.path.splitext(fname)[0]
    with open(os.path.join(SYNC_DIR, fname), "rb") as f:
        data = pickle.load(f)
    frames = torch.stack([transform(Image.open(p).convert("RGB")) for p in data["frames"]]).to(DEVICE)
    audio = torch.tensor(data["audio_features"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        s_feat = sync_model(frames, audio)  # logits or embedding
        s_feat = s_feat.cpu().numpy()
    np.save(os.path.join(OUT_DIR, f"sync_{video_id}.npy"), s_feat)

print("✅ All fusion features saved in:", OUT_DIR)
