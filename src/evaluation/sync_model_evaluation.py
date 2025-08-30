import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from tqdm import tqdm

# ----------------------------- Paths & Setup -----------------------------
SYNC_FEATURE_DIR = "data/processed/sync"
META_DATA_PATH = "data/raw/meta_data.csv"
SYNC_MODEL_PATH = "models/sync_model/sync_model.pth"
PREDICTIONS_DIR = "results/explainability/sync_model_evaluation"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- Model Definition -----------------------------
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
        v_feat = self.cnn(frames)               # [B, 32, H, W]
        a_feat = self.audio_cnn(audio)          # [B, 32, H', W']

        v_feat = v_feat.view(v_feat.size(0), -1)
        a_feat = a_feat.view(a_feat.size(0), -1)

        combined = torch.cat([v_feat, a_feat], dim=1)
        out = self.fc(combined)
        return out

def load_sync_model():
    model = SyncNet().to(DEVICE)
    if os.path.exists(SYNC_MODEL_PATH):
        model.load_state_dict(torch.load(SYNC_MODEL_PATH, map_location=DEVICE))
        print(f"✅ Sync model loaded from {SYNC_MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Sync model not found at {SYNC_MODEL_PATH}")
    model.eval()
    return model

# ----------------------------- Utilities -----------------------------
def load_sync_features(video_id):
    """
    Load synchronized frames and audio features from .pkl or .npz or .npy.
    Adapt as per your actual saved data format.
    """
    sync_path = os.path.join(SYNC_FEATURE_DIR, f"{video_id}.pkl")
    if not os.path.exists(sync_path):
        print(f"⚠️ Sync features not found for {video_id}")
        return None, None
    import pickle
    with open(sync_path, "rb") as f:
        data = pickle.load(f)

    # Load frames images as tensors (dummy example, adjust for your frames loading)
    # Here, you might want to load images and convert to tensor; or your frames could be preprocessed tensors.
    # For demonstration, let's assume frames are paths to jpgs and we load them as RGB tensors:
    import torchvision.transforms as transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    frames_paths = data["frames"]
    frames_tensors = []
    for path in frames_paths:
        img = Image.open(path).convert("RGB")
        frames_tensors.append(transform(img))
    frames_tensor = torch.stack(frames_tensors).permute(1, 0, 2, 3)  # [C, T, H, W]
    frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # back to [T, C, H, W]
    frames_tensor = frames_tensor.mean(dim=0, keepdim=True)  # example to reduce temporal dim for demo
    # Ideally, keep temporal info or sample fixed number of frames

    # Load audio features
    audio_feat = data["audio_features"]  # shape [channels, time]
    audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0)  # [1, channels, time]

    # Reshape audio tensor to fit Conv2d input [B, 1, H, W]
    audio_tensor = audio_tensor.unsqueeze(1)  # [1, 1, channels, time]

    return frames_tensor.to(DEVICE), audio_tensor.to(DEVICE)

def evaluate_sync_model():
    print(">>> Loading sync model...")
    model = load_sync_model()

    print(">>> Loading metadata...")
    df = pd.read_csv(META_DATA_PATH)

    # Filter or sample data as needed; example here uses all data:
    video_ids = df['video_id'].tolist() if 'video_id' in df.columns else df['path'].apply(lambda x: os.path.splitext(x)[0]).tolist()
    labels_map = {'A': 0, 'B': 0, 'C': 1, 'D': 1}  # Real=0, Fake=1 based on your categories

    results = []
    for i, vid in tqdm(enumerate(video_ids), total=len(video_ids)):
        frames, audio = load_sync_features(vid)
        if frames is None or audio is None:
            continue

        true_label = df.loc[df['path'].str.contains(vid), 'category'].values[0]
        true_label = labels_map.get(true_label, 0)

        with torch.no_grad():
            outputs = model(frames, audio)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)

        results.append({
            'video_id': vid,
            'true_label': true_label,
            'predicted': pred_class,
            'prob_real': probs[0],
            'prob_fake': probs[1]
        })

        print(f"[{i}] Video: {vid} True: {true_label} Pred: {pred_class} Prob_fake: {probs[1]:.3f}")

    if not results:
        print("No valid samples for evaluation.")
        return

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(PREDICTIONS_DIR, "sync_predictions.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"✅ Saved predictions to {csv_path}")

    # Metrics and plots
    y_true = df_results['true_label'].values
    y_pred = df_results['predicted'].values
    y_scores = df_results['prob_fake'].values

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    cm_path = os.path.join(PREDICTIONS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(PREDICTIONS_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve saved to {roc_path}")

    # Classification report & accuracy
    report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'])
    accuracy = accuracy_score(y_true, y_pred)

    report_path = os.path.join(PREDICTIONS_DIR, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("Sync Model Evaluation Report\n")
        f.write("="*40 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC: {roc_auc:.4f}\n\n")
        f.write(report)
    print(f"Evaluation report saved to {report_path}")

    print(f"\n🎉 Sync model evaluation done! Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    try:
        evaluate_sync_model()
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
