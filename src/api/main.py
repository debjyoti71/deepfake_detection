from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import tempfile
from src.training.train_fusion import FusionNet  # reuse your fusion model class
import cv2
import librosa

app = FastAPI(title="Deepfake Detection API")

# ----------------------------
# Paths & Device
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VISUAL_MODEL_PATH = "models/visual_model.pth"
AUDIO_MODEL_PATH  = "models/audio_model.pth"
SYNC_MODEL_PATH   = "models/sync_model.pth"
FUSION_MODEL_PATH = "models/fusion_model.pth"

# ----------------------------
# Load Models
# ----------------------------
def load_visual_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(VISUAL_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_audio_model():
    # Simple placeholder CNN for audio
    class AudioCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1,16,3,padding=1)
            self.pool = torch.nn.MaxPool2d(2)
            self.conv2 = torch.nn.Conv2d(16,32,3,padding=1)
            self.pool2 = torch.nn.MaxPool2d(2)
            self.fc = torch.nn.Linear(32*40*40,2)
        def forward(self,x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool2(torch.relu(self.conv2(x)))
            x = x.view(x.size(0),-1)
            return self.fc(x)
    model = AudioCNN()
    model.load_state_dict(torch.load(AUDIO_MODEL_PATH,map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_fusion_model(input_dim):
    model = FusionNet(input_dim)
    model.load_state_dict(torch.load(FUSION_MODEL_PATH,map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# Load models once
visual_model = load_visual_model()
audio_model  = load_audio_model()
fusion_model = None  # will initialize dynamically after extracting features

# ----------------------------
# Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def extract_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = transform(frame)
            frames.append(frame)
        count += 1
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames extracted")
    return torch.stack(frames)

def extract_audio_features(video_path, sr=16000):
    # Extract waveform
    import moviepy.editor as mp
    clip = mp.VideoFileClip(video_path)
    audio = clip.audio.to_soundarray(fps=sr)
    audio = np.mean(audio, axis=1)  # mono
    # Convert to log-mel spectrogram
    import librosa
    mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel)
    log_mel = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 1x1x64xT
    return log_mel

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Extract frames & predict visual
        frames = extract_frames(tmp_path)
        frames = frames.to(DEVICE)
        with torch.no_grad():
            outputs = visual_model(frames)
            visual_probs = torch.softmax(outputs, dim=1)[:,1].mean().item()
            visual_pred = int(visual_probs>=0.5)

        # Extract audio & predict
        audio_feat = extract_audio_features(tmp_path)
        audio_feat = audio_feat.to(DEVICE)
        with torch.no_grad():
            outputs = audio_model(audio_feat)
            audio_probs = torch.softmax(outputs, dim=1)[:,1].mean().item()
            audio_pred = int(audio_probs>=0.5)

        # Fusion prediction (concatenate visual+audio features)
        visual_feat = frames.view(frames.size(0),-1).mean(dim=0)
        audio_feat_vec = audio_feat.view(-1)
        fusion_input = torch.cat([visual_feat,audio_feat_vec],dim=0).unsqueeze(0).to(DEVICE)

        global fusion_model
        if fusion_model is None:
            fusion_model = load_fusion_model(fusion_input.shape[1])
        with torch.no_grad():
            outputs = fusion_model(fusion_input)
            fusion_probs = torch.softmax(outputs, dim=1)[:,1].item()
            fusion_pred = int(fusion_probs>=0.5)

        return JSONResponse({
            "visual_pred": visual_pred,
            "visual_prob": float(visual_probs),
            "audio_pred": audio_pred,
            "audio_prob": float(audio_probs),
            "fusion_pred": fusion_pred,
            "fusion_prob": float(fusion_probs)
        })

    finally:
        os.remove(tmp_path)

# ----------------------------
# Run API
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
