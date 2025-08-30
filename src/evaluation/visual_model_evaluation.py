import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from tqdm import tqdm
from PIL import Image
import cv2

# ----------------------------- Paths & Setup -----------------------------

pred_data_dir = "data/processed/raw"
META_DATA_PATH = "data/raw/meta_data.csv"
PREDICTIONS_DIR = "results/explainability/visual_model_evaluation"
VISUAL_MODEL_PATH = "models/visual_model/visual_model.pth"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- Model Definition -----------------------------

def load_model():
    # Load ResNet18 directly to match saved weights
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    if os.path.exists(VISUAL_MODEL_PATH):
        model.load_state_dict(torch.load(VISUAL_MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded from {VISUAL_MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model not found at {VISUAL_MODEL_PATH}")
    return model

# ----------------------------- Frame Extraction -----------------------------

def extract_frames_from_video(video_path, max_frames=16):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    frame_idxs = np.linspace(0, total_frames - 1, num=max_frames, dtype=int)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    current_idx = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(frame)
            frames.append(tensor)
            current_idx += 1

    cap.release()
    return torch.stack(frames) if frames else None

# ----------------------------- Inference -----------------------------

def predict_video(model, video_tensor):
    model.eval()
    video_tensor = video_tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(video_tensor)  # (N, num_classes)
        probs = torch.softmax(outputs, dim=1)
        avg_probs = probs.mean(dim=0)
        pred_class = torch.argmax(avg_probs).item()
    return pred_class, avg_probs.cpu().numpy()

# ----------------------------- Evaluation Utilities -----------------------------

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def plot_prediction_distribution(y_true, y_scores, save_path):
    plt.figure(figsize=(10, 6))
    
    real_scores = y_scores[y_true == 0]
    fake_scores = y_scores[y_true == 1]
    
    plt.hist(real_scores, bins=30, alpha=0.7, label='Real', color='blue', density=True)
    plt.hist(fake_scores, bins=30, alpha=0.7, label='Fake', color='red', density=True)
    
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_evaluation_report(y_true, y_pred, y_scores, roc_auc, save_path):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], output_dict=True)
    
    with open(save_path, 'w') as f:
        f.write("Visual Model Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total Samples: {len(y_true)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC Score: {roc_auc:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write("-" * 20 + "\n")
        for label, metrics in report.items():
            if label in ['Real', 'Fake']:
                f.write(f"{label}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-score: {metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n\n")
        
        f.write(f"Macro avg:\n")
        f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"  Recall: {report['macro avg']['recall']:.4f}\n")
        f.write(f"  F1-score: {report['macro avg']['f1-score']:.4f}\n")

# ----------------------------- Evaluation Pipeline -----------------------------

def load_metadata():
    return pd.read_csv(META_DATA_PATH)

def evaluate_videos():
    print(">>> Loading model...")
    model = load_model().to(DEVICE)
    
    print(">>> Loading metadata...")
    metadata_df = load_metadata()
    print(f">>> Total metadata rows: {len(metadata_df)}")

    # Filter by race and sample
    filtered_df = metadata_df[metadata_df['race'].isin(["Asian (East)","Asian (South)"])]
    print(f">>> Filtered to {len(filtered_df)} rows with race in ['African', 'Asian (East)']")

    real_df = filtered_df[filtered_df['category'].isin(['A', 'B'])]
    fake_df = filtered_df[filtered_df['category'].isin(['C', 'D'])]
    print(f">>> Found {len(real_df)} real and {len(fake_df)} fake samples before sampling")

    real_sample = real_df.sample(n=min(500, len(real_df)), random_state=42)
    fake_sample = fake_df.sample(n=min(500, len(fake_df)), random_state=42)

    sampled_df = pd.concat([real_sample, fake_sample]).reset_index(drop=True)
    print(f">>> Sampled {len(real_sample)} real and {len(fake_sample)} fake videos")

    results = []

    for i, (_, row) in enumerate(tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Evaluating")):
        video_path = os.path.join(row['full_path'],row['path'])
        if not os.path.exists(video_path):
            print(f"[{i}] ✗ File not found: {video_path}")
            continue

        video_tensor = extract_frames_from_video(video_path)
        if video_tensor is None:
            print(f"[{i}] ✗ Could not extract frames: {video_path}")
            continue
        else:
            print(f"[{i}] ✓ Extracted {video_tensor.shape[0]} frames from {video_path}")

        label = 1 if row['category'] in ['C', 'D'] else 0
        pred_class, prob = predict_video(model, video_tensor)

        print(f"[{i}] Prediction: True={label} | Predicted={pred_class} | Prob=[Real:{prob[0]:.3f}, Fake:{prob[1]:.3f}]")

        results.append({
            'video_path': video_path,
            'category': row['category'],
            'race': row['race'],
            'true_label': label,
            'predicted': pred_class,
            'prob_class_0': prob[0],
            'prob_class_1': prob[1]
        })

    if not results:
        print("✗ No valid videos found for evaluation")
        return

    df_results = pd.DataFrame(results)
    output_csv = os.path.join(PREDICTIONS_DIR, "video_predictions.csv")
    df_results.to_csv(output_csv, index=False)
    print(f">>> ✓ Saved predictions to {output_csv}")

    print(f">>> Processed {len(df_results)} videos")
    print(f">>> Label distribution: Real={sum(df_results['true_label'] == 0)}, Fake={sum(df_results['true_label'] == 1)}")

    y_true = df_results['true_label'].values
    y_pred = df_results['predicted'].values
    y_scores = df_results['prob_class_1'].values

    # Save plots and report
    cm_path = os.path.join(PREDICTIONS_DIR, "confusion_matrix.png")
    roc_path = os.path.join(PREDICTIONS_DIR, "roc_curve.png")
    dist_path = os.path.join(PREDICTIONS_DIR, "prediction_distribution.png")
    report_path = os.path.join(PREDICTIONS_DIR, "evaluation_report.txt")

    print(">>> Generating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    print(f"✓ Saved confusion matrix to {cm_path}")

    print(">>> Generating ROC curve...")
    roc_auc = plot_roc_curve(y_true, y_scores, roc_path)
    print(f"✓ Saved ROC curve to {roc_path}")

    print(">>> Generating prediction distribution plot...")
    plot_prediction_distribution(np.array(y_true), np.array(y_scores), dist_path)
    print(f"✓ Saved prediction distribution plot to {dist_path}")

    print(">>> Saving text report...")
    save_evaluation_report(y_true, y_pred, y_scores, roc_auc, report_path)
    print(f"✓ Saved report to {report_path}")

    print("\n🎉 Evaluation complete!")
    print(f"✔ Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"✔ AUC: {roc_auc:.4f}")
    print(f"✔ Results saved to: {PREDICTIONS_DIR}")

# ----------------------------- Entry Point -----------------------------

if __name__ == "__main__":
    try:
        evaluate_videos()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
