# 🐦 BirdVision India

**A Real-Time Multimodal Bird Detection System for Conservation Monitoring in India**

## 📌 Project Overview

**BirdVision India** is a research-driven multimodal AI system designed to detect and classify Indian bird species using both **audio and visual inputs**. Developed as part of the M.Sc. Big Data Analytics program at **St. Xavier’s College, Mumbai**, the project integrates state-of-the-art models like **EfficientNetB0** (for audio classification via spectrograms) and **YOLOv8** (for object detection in images and videos).

The system aims to support **biodiversity monitoring, ecological research, and conservation efforts** by enabling real-time bird species recognition through a web-based interface.

---

## 🎯 Key Features

- 🧠 **Multimodal Detection**: Combines real-time image and audio analysis.
- 🎙️ **Audio Detection**: Classifies bird calls using Mel-spectrograms and a CNN model (EfficientNetB0).
- 📷 **Visual Detection**: Detects birds in images/videos using YOLOv8n with EfficientNet-lite backbone.
- 🔄 **Max Voting Strategy**: Robust prediction from chunked audio.
- 🧭 **Interactive Web App**: Built with Flask and supports uploads + live webcam/mic inputs.
- 🌍 **Conservation-Centric**: Focused on 8 Indian bird species selected based on the *State of India’s Birds 2023* report.
│   ├── images/
│   └── audio/
├── notebooks/                # Jupyter notebooks for experiments
├── requirements.txt
├── README.md
└── LICENSE
