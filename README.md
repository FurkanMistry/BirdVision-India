# 🐦 BirdVision India

**A Real-Time Multimodal Bird Detection System for Conservation Monitoring in India**

## 📌 Project Overview

**BirdVision India** is a multimodal AI system designed to detect and classify Indian bird species using both **audio and visual inputs**. Developed as part of the M.Sc. Big Data Analytics program at **St. Xavier’s College, Mumbai**, this system leverages **EfficientNetB0** for audio classification and **YOLOv8n** for visual object detection.

It aids in ecological research and biodiversity monitoring through a web-based application that allows users to upload images, videos, or bird calls and receive species predictions in real time.

---

## 🎯 Key Features

- 🔊 **Audio Detection**: Classifies bird calls using Mel-spectrograms and EfficientNetB0.
- 📸 **Visual Detection**: Identifies birds in images and videos using YOLOv8n + EfficientNet-lite.
- 🤖 **Multimodal Fusion**: Combines audio and image detections for improved robustness.
- ⚡ **Real-time Inference**: Live webcam and microphone detection support.
- 🌱 **Conservation Focus**: Trained on 8 Indian species (high-priority & ecological indicators).
- 🌐 **Flask Web App**: Lightweight UI for uploads and live predictions.

---

## 🐦 Bird Species Covered

| ID | Species Name             | Category               |
|----|--------------------------|------------------------|
| 0  | Asian Koel               | Environment Indicator |
| 1  | Black Kite               | Environment Indicator |
| 2  | Common Kingfisher        | Environment Indicator |
| 3  | Common Myna              | High-Priority         |
| 4  | House Sparrow            | Environment Indicator |
| 5  | Little Ringed Plover     | High-Priority         |
| 6  | Rose-Ringed Parakeet     | Environment Indicator |
| 7  | Whiskered Tern           | High-Priority         |

---

## 🔧 Tech Stack

- **Python 3.10**
- **TensorFlow / Keras** – Audio model
- **YOLOv8 (Roboflow)** – Image/video detection
- **Librosa** – Audio processing
- **OpenCV** – Real-time frame analysis
- **Flask** – Web backend
- **Tailwind CSS** – Responsive frontend
- **Supervision (sv)** – Detection annotation

---

## 📈 Performance

### 🎙️ Audio Model

- Accuracy: **89.02% (Test)**, **96.86% (Validation)**
- F1-Score: **0.86**
- Strategy: Mel-spectrogram + chunk-wise inference + max voting

### 📷 Visual Model

- mAP@50: **90.6%**
- Precision: **90.8%**
- Recall: **85.9%**

---

### Folder Structure
BirdVision-India/
├── static/                          # Static assets (CSS, JS, images)
├── templates/                       # HTML templates for Flask
├── uploads/                         # Temporary file uploads
├── 8birds_final.ipynb              # Jupyter notebook for model training/testing
├── BirdVision India.pbix           # Power BI report (dashboard/EDA)
├── BirdVision_TestCases.xlsx       # Test cases (audio/image/multimodal)
├── README.md                       # Project README file
├── app.py                          # Flask app backend
├── map.py                          # Possibly used for geo-visualization
├── model_16_val_loss_0.1970.keras  # Trained EfficientNetB0 audio model
├── train_metadata - Copy.csv       # Metadata used for training/testing
