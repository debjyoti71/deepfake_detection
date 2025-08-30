import os
import glob
import cv2
import numpy as np
import onnxruntime as ort

# Paths
FRAMES_DIR = "data/processed/frames"
OUTPUT_DIR = "data/processed/faces"
MODEL_PATH = "yolov8_face.onnx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize ONNX Runtime (DirectML = AMD GPU)
providers = ['DmlExecutionProvider']  # This enables AMD GPU usage
session = ort.InferenceSession(MODEL_PATH, providers=providers)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape[2:]  # (H, W)

def preprocess(img):
    resized = cv2.resize(img, input_shape[::-1])
    blob = resized / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return blob

def postprocess(output, original_shape, conf_threshold=0.5):
    detections = output[0]
    boxes = []
    for det in detections:
        conf = det[4]
        if conf >= conf_threshold:
            x_center, y_center, width, height = det[:4]
            x1 = int((x_center - width / 2) * original_shape[1])
            y1 = int((y_center - height / 2) * original_shape[0])
            x2 = int((x_center + width / 2) * original_shape[1])
            y2 = int((y_center + height / 2) * original_shape[0])
            boxes.append((x1, y1, x2, y2))
    return boxes

def extract_faces_from_frame(frame_path, output_path):
    img = cv2.imread(frame_path)
    if img is None:
        print(f"❌ Could not read {frame_path}")
        return False

    input_blob = preprocess(img)
    outputs = session.run(None, {input_name: input_blob})
    boxes = postprocess(outputs, img.shape)

    if not boxes:
        return False

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        face = img[max(0, y1):y2, max(0, x1):x2]
        if face.size == 0:
            continue
        face = cv2.resize(face, (224, 224))
        face_filename = os.path.join(
            output_path, f"{os.path.splitext(os.path.basename(frame_path))[0]}_face{i}.jpg"
        )
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
