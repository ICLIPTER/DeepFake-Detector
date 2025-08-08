import cv2
import os
from mtcnn import MTCNN

# Paths
RAW_VIDEOS_DIR = "../data/raw_videos"
OUTPUT_DIR = "../data/processed_frames"

# Settings
FRAME_INTERVAL = 10  # extract every Nth frame
IMG_SIZE = (160, 160)  # size for model input

detector = MTCNN()

def extract_faces_from_video(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_frame)

            for i, face in enumerate(faces):
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                cropped_face = frame[y:y + h, x:x + w]

                if cropped_face.size > 0:
                    resized_face = cv2.resize(cropped_face, IMG_SIZE)
                    save_path = os.path.join(output_folder, f"frame_{saved_count:03d}.jpg")
                    cv2.imwrite(save_path, resized_face)
                    saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[INFO] {saved_count} faces saved to {output_folder}")

def process_all_videos():
    for label in ["real", "fake"]:
        input_dir = os.path.join(RAW_VIDEOS_DIR, label)
        for file_name in os.listdir(input_dir):
            if file_name.lower().endswith(".mp4"):
                video_path = os.path.join(input_dir, file_name)
                video_name = os.path.splitext(file_name)[0]
                output_folder = os.path.join(OUTPUT_DIR, label, video_name)
                print(f"[INFO] Processing {video_path}...")
                extract_faces_from_video(video_path, output_folder)

if __name__ == "__main__":
    process_all_videos()
