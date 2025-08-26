import os
import glob
from mtcnn import MTCNN
import cv2

# Input frames and output face crops
FRAMES_DIR = "data/processed/frames"
OUTPUT_DIR = "data/processed/faces"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize face detector
detector = MTCNN()

def extract_faces_from_frame(frame_path, output_path):
    """Detect and crop faces from a single frame."""
    img = cv2.imread(frame_path)
    if img is None:
        print(f"❌ Could not read {frame_path}")
        return False

    results = detector.detect_faces(img)

    if len(results) == 0:
        return False  # No face found

    for i, res in enumerate(results):
        x, y, w, h = res['box']
        x, y = max(0, x), max(0, y)
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))  # Resize for model input
        face_filename = os.path.join(output_path, os.path.basename(frame_path))
        cv2.imwrite(face_filename, face)

    return True

def process_frames():
    video_dirs = glob.glob(os.path.join(FRAMES_DIR, "*"))

    for video_dir in video_dirs:
        video_id = os.path.basename(video_dir)
        output_subdir = os.path.join(OUTPUT_DIR, video_id)
        os.makedirs(output_subdir, exist_ok=True)

        frame_files = glob.glob(os.path.join(video_dir, "*.jpg"))
        print(f"📂 Processing {video_id} ({len(frame_files)} frames)")

        for frame_path in frame_files:
            extract_faces_from_frame(frame_path, output_subdir)

        print(f"✅ Done: {video_id}")

if __name__ == "__main__":
    process_frames()
