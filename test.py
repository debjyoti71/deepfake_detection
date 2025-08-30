import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from tqdm import tqdm
import random


pred_data_dir = "data/processed/raw"
META_DATA_PATH = "data/raw/meta_data.csv"
PREDICTIONS_DIR = "results/explainability/visual_model_evaluation"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
VISUAL_MODEL_PATH = "models/visual_model.pth/visual_model.pth"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualModel(nn.Module):
    def __init__(self, num_classes=2):
        super(VisualModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def load_model():
    model = VisualModel()
    if os.path.exists(VISUAL_MODEL_PATH):
        model.load_state_dict(torch.load(VISUAL_MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded from {VISUAL_MODEL_PATH}")
    else:
        print(f"Model not found at {VISUAL_MODEL_PATH}")


def load_metadata():
    df = pd.read_csv(META_DATA_PATH)
    return df


def load_videos():
    metadata_df = load_metadata()
    sampled_df = metadata_df.sample(n=min(1000, len(metadata_df)), random_state=42)

    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Evaluating"):
            if row['race'] == ["African","Asian (East)"]:
                video_path = row['path']
                video_name = row['filename']
                label = 1 if row['category'] in ['C', 'D'] else 0
            
            # Look for face images from this video
            pred_file = os.path.join(video_path,video_name)
            if not os.path.exists(pred_file):
                continue
