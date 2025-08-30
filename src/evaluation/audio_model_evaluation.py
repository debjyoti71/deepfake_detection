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
FEATURE_DIR = "data/processed/audio_features"
META_DATA_PATH = "data/raw/meta_data.csv"
PREDICTIONS_DIR = "results/explainability/audio_model_evaluation"
AUDIO_MODEL_PATH = "models/audio_model/audio_model.pth"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- Model Definition -----------------------------
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
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def load_model():
    model = AudioCNN().to(DEVICE)
    if os.path.exists(AUDIO_MODEL_PATH):
        model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=DEVICE))
        print(f"✅ Model loaded from {AUDIO_MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model not found at {AUDIO_MODEL_PATH}")
    return model

# ----------------------------- Inference -----------------------------
def load_audio_feature(feature_path):
    try:
        mel = np.load(feature_path)
        target_len = 512
        if mel.shape[1] < target_len:
            pad_width = target_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
        elif mel.shape[1] > target_len:
            mel = mel[:, :target_len]
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return mel_tensor
    except Exception as e:
        print(f"❌ Failed to load {feature_path}: {e}")
        return None

def predict_audio(model, mel_tensor):
    model.eval()
    mel_tensor = mel_tensor.to(DEVICE)
    with torch.no_grad():
        output = model(mel_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs.squeeze().cpu().numpy()

# ----------------------------- Evaluation Utilities -----------------------------
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return roc_auc

def plot_prediction_distribution(y_true, y_scores, save_path):
    plt.figure(figsize=(10, 6))
    real_scores = y_scores[y_true == 0]
    fake_scores = y_scores[y_true == 1]
    plt.hist(real_scores, bins=30, alpha=0.6, label="Real", color="blue", density=True)
    plt.hist(fake_scores, bins=30, alpha=0.6, label="Fake", color="red", density=True)
    plt.xlabel("Probability of Fake")
    plt.ylabel("Density")
    plt.title("Prediction Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_report(y_true, y_pred, y_scores, roc_auc, save_path):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], output_dict=True)
    with open(save_path, 'w') as f:
        f.write("Audio Model Evaluation Report\n")
        f.write("="*40 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC Score: {roc_auc:.4f}\n\n")
        for label in ['Real', 'Fake']:
            metrics = report[label]
            f.write(f"{label}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1-score']:.4f}\n\n")
        macro = report['macro avg']
        f.write("Macro Average:\n")
        f.write(f"  Precision: {macro['precision']:.4f}\n")
        f.write(f"  Recall:    {macro['recall']:.4f}\n")
        f.write(f"  F1-Score:  {macro['f1-score']:.4f}\n")

# ----------------------------- Evaluation Pipeline -----------------------------
def evaluate_audio():
    print(">>> Loading model...")
    model = load_model()

    print(">>> Loading metadata...")
    df = pd.read_csv(META_DATA_PATH)
    df = df[df['race'].isin(["Asian (East)", "Asian (South)"])]
    real_df = df[df['category'].isin(['A', 'B'])]
    fake_df = df[df['category'].isin(['C', 'D'])]

    real_sample = real_df.sample(n=min(500, len(real_df)), random_state=42)
    fake_sample = fake_df.sample(n=min(500, len(fake_df)), random_state=42)
    sample_df = pd.concat([real_sample, fake_sample]).reset_index(drop=True)
    print(f">>> Using {len(sample_df)} total samples (Real: {len(real_sample)}, Fake: {len(fake_sample)})")

    results = []

    for i, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Evaluating"):
        video_id = os.path.splitext(row['path'])[0]
        feature_path = os.path.join(FEATURE_DIR, f"{video_id}.npy")

        if not os.path.exists(feature_path):
            print(f"[{i}] ⚠️ Feature not found: {feature_path}")
            continue

        mel_tensor = load_audio_feature(feature_path)
        if mel_tensor is None:
            print(f"[{i}] ⚠️ Skipped due to loading error")
            continue

        label = 1 if row['category'] in ['C', 'D'] else 0
        pred_class, probs = predict_audio(model, mel_tensor)
        print(f"[{i}] ✓ True: {label} | Pred: {pred_class} | Prob[Real]: {probs[0]:.3f}, Prob[Fake]: {probs[1]:.3f}")

        results.append({
            'video_id': video_id,
            'true_label': label,
            'predicted': pred_class,
            'prob_class_0': probs[0],
            'prob_class_1': probs[1],
            'race': row['race'],
            'category': row['category']
        })

    if not results:
        print("✗ No valid features for evaluation.")
        return

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(PREDICTIONS_DIR, "audio_predictions.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"✅ Saved predictions to {csv_path}")

    y_true = df_results['true_label'].values
    y_pred = df_results['predicted'].values
    y_scores = df_results['prob_class_1'].values

    # Visualizations
    cm_path = os.path.join(PREDICTIONS_DIR, "confusion_matrix.png")
    roc_path = os.path.join(PREDICTIONS_DIR, "roc_curve.png")
    dist_path = os.path.join(PREDICTIONS_DIR, "prediction_distribution.png")
    report_path = os.path.join(PREDICTIONS_DIR, "evaluation_report.txt")

    plot_confusion_matrix(y_true, y_pred, cm_path)
    print(f"✅ Confusion matrix saved to {cm_path}")

    auc_score = plot_roc_curve(y_true, y_scores, roc_path)
    print(f"✅ ROC curve saved to {roc_path}")

    plot_prediction_distribution(np.array(y_true), np.array(y_scores), dist_path)
    print(f"✅ Prediction score distribution saved to {dist_path}")

    save_report(y_true, y_pred, y_scores, auc_score, report_path)
    print(f"✅ Report saved to {report_path}")

    print("\n🎉 Audio model evaluation complete!")
    print(f"✔ Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"✔ AUC: {auc_score:.4f}")
    print(f"✔ Results saved in: {PREDICTIONS_DIR}")

# ----------------------------- Entry Point -----------------------------
if __name__ == "__main__":
    try:
        evaluate_audio()
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
