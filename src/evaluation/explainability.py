import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Function
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# ----------------------------
# Paths
# ----------------------------
FACES_DIR = "data/processed/faces"
PREDICTIONS_DIR = "results/explainability"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
VISUAL_MODEL_PATH = "models/visual_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Grad-CAM Implementation
# ----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        x = x.unsqueeze(0).to(DEVICE)
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        loss = output[0,class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (x.size(3), x.size(2)))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize
        return cam

# ----------------------------
# Load Visual Model
# ----------------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features,2)
model.load_state_dict(torch.load(VISUAL_MODEL_PATH,map_location=DEVICE))
model.to(DEVICE).eval()

# Use last convolutional layer
target_layer = model.layer4[-1]

gradcam = GradCAM(model, target_layer)

# ----------------------------
# Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----------------------------
# Generate Grad-CAM for test videos
# ----------------------------
for video_id in os.listdir(FACES_DIR):
    video_path = os.path.join(FACES_DIR, video_id)
    if not os.path.isdir(video_path):
        continue
    out_dir = os.path.join(PREDICTIONS_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])
    for f in tqdm(frame_files, desc=f"Processing {video_id}"):
        img_path = os.path.join(video_path,f)
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)
        cam = gradcam.generate(tensor)

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed = np.array(img) * 0.5 + heatmap * 0.5
        superimposed = np.uint8(superimposed)

        out_path = os.path.join(out_dir,f"cam_{f}")
        cv2.imwrite(out_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
