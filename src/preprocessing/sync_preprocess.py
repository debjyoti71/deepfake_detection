import os
import numpy as np
import cv2
import librosa
import pickle
from tqdm import tqdm

FRAMES_DIR = "data/processed/frames"
AUDIO_FEAT_DIR = "data/processed/audio_features"
SYNC_OUT_DIR = "data/processed/sync"
os.makedirs(SYNC_OUT_DIR, exist_ok=True)

def align_frames_audio(video_id, fps=25, hop_length=160, sr=16000):
    """
    Align video frames with audio features using timestamps.
    - fps: video frames per second (usually 25 or 30)
    - hop_length: step size of mel spectrogram (10ms -> 160 samples at 16kHz)
    """

    # 1. Load frames
    frame_dir = os.path.join(FRAMES_DIR, video_id)
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])

    # 2. Load audio features
    audio_feat_path = os.path.join(AUDIO_FEAT_DIR, f"{video_id}.npy")
    audio_features = np.load(audio_feat_path)

    # 3. Compute time stamps
    n_frames = len(frames)
    video_duration = n_frames / fps
    n_audio_frames = audio_features.shape[1]
    audio_duration = (n_audio_frames * hop_length) / sr

    # 4. Align by min duration
    min_duration = min(video_duration, audio_duration)
    n_video_sync = int(min_duration * fps)
    n_audio_sync = int(min_duration * sr / hop_length)

    frames = frames[:n_video_sync]
    audio_features = audio_features[:, :n_audio_sync]

    # 5. Package alignment
    sync_data = {
        "frames": [os.path.join(frame_dir, f) for f in frames],
        "audio_features": audio_features,
        "fps": fps,
        "sr": sr
    }

    return sync_data

def process_all():
    video_ids = [d for d in os.listdir(FRAMES_DIR) if os.path.isdir(os.path.join(FRAMES_DIR, d))]
    for vid in tqdm(video_ids):
        try:
            sync_data = align_frames_audio(vid)
            out_path = os.path.join(SYNC_OUT_DIR, f"{vid}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(sync_data, f)
            print(f"Saved sync dataset: {out_path}")
        except Exception as e:
            print(f"[ERROR] {vid}: {e}")

if __name__ == "__main__":
    process_all()
