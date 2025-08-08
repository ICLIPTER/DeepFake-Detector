import os
import pickle
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model

# ==== SETTINGS ====
MODEL_PATH = "../models/deepfake_cnn_lstm.h5"
LABELS_PATH = "../models/labels.pkl"
IMG_SIZE = (128, 128)
SEQUENCE_LENGTH = 20

# ==== LOAD MODEL & LABELS ====
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "rb") as f:
    label_map = pickle.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# ==== FUNCTION TO EXTRACT FRAMES ====
def extract_frames_from_video(video_path, sequence_length=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // sequence_length, 1)

    for i in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

    cap.release()
    return np.array(frames)

# ==== DETECTION ====
video_path = input("Enter path to video: ").strip()

if not os.path.exists(video_path):
    print("[ERROR] Video file not found.")
    exit()

frames = extract_frames_from_video(video_path)

if frames.shape[0] != SEQUENCE_LENGTH:
    print("[ERROR] Not enough frames extracted for prediction.")
    exit()

frames = np.expand_dims(frames, axis=0)  # Add batch dimension

prediction = model.predict(frames)[0][0]
pred_class = int(round(prediction))

print(f"[RESULT] This video is predicted as: {inv_label_map[pred_class]} (confidence: {prediction:.4f})")
