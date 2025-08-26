import cv2
import os
import glob
from tqdm import tqdm
import multiprocessing

RAW_DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed/frames"
FRAME_SKIP = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_frames(video_path_output):
    """Extract frames from a single video every N frames."""
    video_path, output_subdir = video_path_output
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error opening {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    frame_id = 0

    os.makedirs(output_subdir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_SKIP == 0:
            frame_filename = os.path.join(output_subdir, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_id += 1

        count += 1

    cap.release()

def process_videos_parallel():
    video_files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.mp4"), recursive=True)

    # Prepare (video_path, output_subdir) tuples
    video_tasks = []
    for video_path in video_files:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        output_subdir = os.path.join(OUTPUT_DIR, video_id)
        video_tasks.append((video_path, output_subdir))

    # Use multiprocessing Pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(extract_frames, video_tasks), total=len(video_tasks), desc="Processing videos"))

if __name__ == "__main__":
    process_videos_parallel()
