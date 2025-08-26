import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed/audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_audio_from_video(video_path):
    from moviepy import VideoFileClip  # Import inside for multiprocessing compatibility

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}.wav")

    if os.path.exists(output_path):
        return f"⏭️ Skipped (already exists): {output_path}"
    
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            return f"⚠️ No audio in {video_path}"
        clip.audio.write_audiofile(output_path, codec="pcm_s16le", logger=None)
        return None  # Don't print anything on success
    except Exception as e:
        return f"❌ Error processing {video_path}: {e}"

def process_all_videos():
    video_files = glob.glob(os.path.join(INPUT_DIR, "**/*.mp4"), recursive=True)
    print(f"🎥 Found {len(video_files)} videos")

    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(extract_audio_from_video, video_files, chunksize=4), total=len(video_files)):
            if result:  # Only print warnings or errors
                print(result)

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    process_all_videos()
