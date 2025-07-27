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

## 📁 Project Structure

```bash
BirdVision-India/
├── static/                          # Static assets (CSS, JS, images)
│   ├── style.css                    # Custom styling
│   ├── audio.js                     # Audio detection interface
│   ├── image_video.js              # Image/video upload interface
│   └── uploads/                     # Temporary file uploads
├── templates/                       # HTML templates for Flask
│   ├── base.html                   # Base template
│   ├── home.html                   # Home page
│   ├── audio.html                  # Audio detection page
│   ├── image_video.html           # Image/video detection page
│   └── realtime.html              # Real-time detection page
├── config.py                       # Configuration settings
├── requirements.txt               # Python dependencies
├── app.py                         # Flask application backend
├── map.py                         # Map visualization script
├── model_16_val_loss_0.1970.keras # Trained EfficientNetB0 audio model
├── train_metadata - Copy.csv     # Training/test metadata
├── 8birds_final.ipynb           # Model training notebook
├── BirdVision India.pbix        # Power BI dashboard
├── BirdVision_TestCases.xlsx    # Manual test cases
└── README.md                    # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Audio/video codecs for your system
- Webcam and microphone (for real-time detection)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/FurkanMistry/BirdVision-India.git
   cd BirdVision-India
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   # Create .env file or set environment variables
   export ROBOFLOW_API_KEY="your-roboflow-api-key"
   export SECRET_KEY="your-secret-key"
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Access the application:**
   Open http://localhost:5000 in your browser

## 🔧 Configuration

The application uses a centralized configuration system in `config.py`. Key settings:

- **API Keys**: Set `ROBOFLOW_API_KEY` environment variable
- **Model Paths**: Configure paths to trained models
- **Detection Thresholds**: Adjust confidence thresholds for audio/visual detection
- **File Limits**: Configure upload size limits and allowed extensions

## 🎯 Usage Guide

### Audio Detection
1. Navigate to the "Audio Detection" page
2. Upload an audio file (.wav, .mp3, .ogg, .flac)
3. View mel-spectrogram visualizations and predictions
4. Get confidence scores and top species matches

### Image/Video Detection  
1. Go to "Image/Video Detection" page
2. Upload image (.png, .jpg, .jpeg, .gif) or video (.mp4, .avi, .mov)
3. View annotated results with bounding boxes and species labels
4. Download processed images with detections

### Real-time Detection
1. Visit "Real-Time Detection" page
2. Grant camera and microphone permissions
3. Start detection to see live multimodal analysis
4. Combines visual object detection with audio classification

## 🧪 Testing

Run manual test cases documented in `BirdVision_TestCases.xlsx`:
- Audio file compatibility tests
- Image/video processing validation
- Real-time detection accuracy
- Error handling verification

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **St. Xavier's College, Mumbai** - M.Sc. Big Data Analytics Program
- **Roboflow** - Vision model training platform
- **TensorFlow/Keras** - Deep learning framework
- **Librosa** - Audio processing library
