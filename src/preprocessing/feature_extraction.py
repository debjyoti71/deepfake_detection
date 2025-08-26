import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm

# Paths
AUDIO_DIR = "data/processed/audio"
FEATURE_DIR = "data/processed/audio_features"
os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_mel_spectrogram(audio_path, sr=16000, n_mels=80, hop_length=160, win_length=400):
    y, _ = librosa.load(audio_path, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=512, hop_length=hop_length, 
        win_length=win_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # convert to log scale
    return mel_db

def process_all_audios():
    for fname in tqdm(os.listdir(AUDIO_DIR)):
        if fname.endswith(".wav"):
            audio_path = os.path.join(AUDIO_DIR, fname)
            video_id = os.path.splitext(fname)[0]

            # Extract features
            mel = extract_mel_spectrogram(audio_path)

            # Save as numpy
            out_path = os.path.join(FEATURE_DIR, f"{video_id}.npy")
            np.save(out_path, mel)

            print(f"Saved: {out_path}")

if __name__ == "__main__":
    process_all_audios()
