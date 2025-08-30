import os
import numpy as np
import pickle
from tqdm import tqdm

# Directories
FRAMES_DIR = "data/processed/frames"
AUDIO_FEAT_DIR = "data/processed/audio_features"
SYNC_OUT_DIR = "data/processed/sync"

os.makedirs(SYNC_OUT_DIR, exist_ok=True)

def align_frames_audio(video_id, fps=25, hop_length=160, sr=16000):
    """
    Align video frames with audio features based on duration.

    Args:
        video_id (str): Video identifier (folder name).
        fps (int): Frames per second of video.
        hop_length (int): Number of audio samples between mel frames.
        sr (int): Audio sampling rate.

    Returns:
        dict: Dictionary with keys:
            - 'frames': list of aligned frame file paths
            - 'audio_features': numpy array of aligned audio features
            - 'fps': video fps
            - 'sr': audio sampling rate
    """

    frame_dir = os.path.join(FRAMES_DIR, video_id)
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"Frames directory not found for video {video_id}: {frame_dir}")

    # Load and sort frame filenames
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    if len(frames) == 0:
        raise ValueError(f"No frames found in directory {frame_dir}")

    # Load audio features
    audio_feat_path = os.path.join(AUDIO_FEAT_DIR, f"{video_id}.npy")
    if not os.path.exists(audio_feat_path):
        raise FileNotFoundError(f"Audio feature file not found: {audio_feat_path}")

    audio_features = np.load(audio_feat_path)

    # Calculate durations
    n_frames = len(frames)
    video_duration = n_frames / fps
    n_audio_frames = audio_features.shape[1]
    audio_duration = (n_audio_frames * hop_length) / sr

    # Align lengths based on minimum duration
    min_duration = min(video_duration, audio_duration)
    n_video_sync = int(min_duration * fps)
    n_audio_sync = int(min_duration * sr / hop_length)

    # Trim frames and audio features to aligned length
    frames = frames[:n_video_sync]
    audio_features = audio_features[:, :n_audio_sync]

    # Build full frame paths
    frame_paths = [os.path.join(frame_dir, f) for f in frames]

    return {
        "frames": frame_paths,
        "audio_features": audio_features,
        "fps": fps,
        "sr": sr
    }

def process_all():
    """
    Process all videos in the frames directory,
    aligning frames and audio features and saving as pickle files.
    """
    video_ids = [d for d in os.listdir(FRAMES_DIR) if os.path.isdir(os.path.join(FRAMES_DIR, d))]
    print(f"Found {len(video_ids)} videos to process.")

    for vid in tqdm(video_ids, desc="Processing videos"):
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
