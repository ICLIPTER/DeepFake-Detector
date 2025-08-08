<h1>DeepFake-Detector</h1>

![DeepFake-Detector](https://your-image-link-if-any)  
*Accurately detect deepfake videos using advanced deep learning techniques.*

---

## Overview

DeepFake-Detector is a Python-based deep learning project designed to detect whether a video is real or fake (deepfake). It uses a CNN-LSTM architecture with EfficientNet as the backbone to extract features from video frames and classifies them with high accuracy.

This project also includes a Flask web app interface to upload videos and get real-time predictions.

---

## Features

- Face extraction from videos  
- Deepfake detection using CNN + LSTM model  
- User-friendly Flask web interface for video upload and prediction  
- Supports batch processing and real-time video analysis  
- Modular, extensible codebase for further research and improvements

---

## Project Structure 
DeepFake-Detector/
├── data/
│   ├── raw_videos/
│   │   ├── real/
│   │   └── fake/
│   └── processed_frames/
├── models/
│   ├── deepfake_cnn_lstm.h5
│   └── labels.pkl
├── scripts/
│   ├── extract_faces_auto.py
│   ├── train_model.py
│   └── detect_video.py
├── webapp/
│   ├── app.py
│   ├── templates/
│   └── static/
├── requirements.txt
└── README.md

## Demo

To try the web interface locally:

``bash
cd webapp
pip install -r requirements.txt
python app.py
``

Installation
1 Prerequisites
2 Python 3.8+

pip package manager
Setup

Clone the repository:
git clone https://github.com/ICLIPTER/DeepFake-Detector.git
cd DeepFake-Detector

(Optional but recommended) Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate        # Windows

Install dependencies:
pip install -r requirements.txt

Run Web Application
cd webapp
python app.py




