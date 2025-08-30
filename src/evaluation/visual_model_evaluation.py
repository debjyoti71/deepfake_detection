import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Function
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

FACES_DIR = "data/raw"
PREDICTIONS_DIR = "results/explainability/visual_model_evaluation"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
VISUAL_MODEL_PATH = "models/visual_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


