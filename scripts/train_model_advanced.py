import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# ===========================
# CONFIG
# ===========================
DATA_DIR = "../data/processed_frames"
MODEL_PATH = "../models/deepfake_cnn_lstm_advanced.h5"
LABELS_PATH = "../models/labels.pkl"
SEQUENCE_LENGTH = 20  # more frames per video
IMG_SIZE = (128, 128)

# ===========================
# LOAD DATA
# ===========================
def load_data():
    X, y = [], []
    labels = sorted(os.listdir(DATA_DIR))
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        label_dir = os.path.join(DATA_DIR, label)
        for video_folder in os.listdir(label_dir):
            vid_path = os.path.join(label_dir, video_folder)

            if not os.path.isdir(vid_path):
                continue

            frame_files = sorted(os.listdir(vid_path))[:SEQUENCE_LENGTH]
            frames = []
            for frame_file in frame_files:
                frame_path = os.path.join(vid_path, frame_file)
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0
                frames.append(img)

            if len(frames) == SEQUENCE_LENGTH:
                X.append(frames)
                y.append(label_map[label])

    return np.array(X), np.array(y), label_map

print("[INFO] Loading processed frames...")
X, y, label_map = load_data()
print(f"[INFO] Dataset size: {len(X)} samples")

# Save labels
os.makedirs("../models", exist_ok=True)
with open(LABELS_PATH, "wb") as f:
    pickle.dump(label_map, f)

# One-hot encode labels
y = to_categorical(y, num_classes=len(label_map))

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
# DATA AUGMENTATION
# ===========================
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# ===========================
# MODEL
# ===========================
model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)),
    TimeDistributed(MaxPooling2D(2, 2)),
    TimeDistributed(Dropout(0.25)),

    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(2, 2)),
    TimeDistributed(Dropout(0.25)),

    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ===========================
# CALLBACKS
# ===========================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')

# ===========================
# TRAIN
# ===========================
print("[INFO] Training model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=4),
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stop, checkpoint]
)

print(f"[INFO] Best model saved to {MODEL_PATH}")
