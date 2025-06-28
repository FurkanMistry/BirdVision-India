from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import supervision as sv
from inference.models.utils import get_roboflow_model
import numpy as np
import pyaudio
import threading
import tensorflow as tf
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import matplotlib
import base64
import io
from werkzeug.utils import secure_filename
import plotly.express as px
import pandas as pd
matplotlib.use('Agg')

app = Flask(__name__)

# --- CONFIG ---
USE_WEBCAM = True
VIDEO_PATH = "BK.mp4"
CONFIDENCE_THRESHOLD = 0.78
RATE = 22050
CHANNELS = 1
AUDIO_FORMAT = pyaudio.paFloat32
CHUNK_DURATION = 5
AUDIO_CHUNK = RATE * CHUNK_DURATION

# Global variables
video_running = False
model = load_model('model_16_val_loss_0.1970.keras')
vision_model = get_roboflow_model(
        model_id="birdvision-india/4",
        api_key=api_key
    )

# Bird class dictionary
class_dict = {
    0: 'Asian Koel',
    1: 'Black Kite',
    2: 'Common Kingfisher',
    3: 'Common Myna',
    4: 'House Sparrow',
    5: 'Little Ringed Plover',
    6: 'Rose Ringed Paraket',
    7: 'Whiskered Tern'
}

# Audio label map for /predict endpoint
label_map = {
    0: 'Asian Koel',
    1: 'Black Kite',
    2: 'Common Kingfisher',
    3: 'Common Myna',
    4: 'House Sparrow',
    5: 'Little Ringed Plover',
    6: 'Rose Ringed Paraket',
    7: 'Whiskered Tern'
}

# Image/Video detection logic (from provided code)
UPLOAD_FOLDER_IMG = os.path.join('static', 'uploads')
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_IMG
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(UPLOAD_FOLDER_IMG, exist_ok=True)

# Initialize the vision model (if not already)
if 'vision_model' not in globals() or vision_model is None:
    vision_model = get_roboflow_model(
        model_id="birdvision-india/4",
        api_key="SwZPL9CiE1cRQ5i2YS7C"
    )

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

class AudioProcessor(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK
        )
        self.running = True
        self.latest_prediction = None
        self.bird_name = None
        self.confidence = 0.0
        self.lock = threading.Lock()

    def process_audio_chunk(self, audio_data):
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=RATE,
            n_mels=224,
            hop_length=512
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.figure(figsize=(2.24, 2.24), dpi=100)
            plt.axis('off')
            librosa.display.specshow(mel_spectrogram_db, sr=RATE, hop_length=512, cmap='magma')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(tmpfile.name, bbox_inches='tight', pad_inches=0)
            plt.close()
            img = Image.open(tmpfile.name).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def run(self):
        global model
        while self.running:
            try:
                audio_data = np.frombuffer(
                    self.stream.read(AUDIO_CHUNK, exception_on_overflow=False),
                    dtype=np.float32
                )
                img_array = self.process_audio_chunk(audio_data)
                prediction = model.predict(img_array, verbose=0)
                self.latest_prediction = prediction
                predicted_class = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class])
                if confidence > 0.85:
                    with self.lock:
                        self.bird_name = class_dict.get(predicted_class, "Unknown")
                        self.confidence = confidence
                    print(f"Audio Detection: {self.bird_name} ({self.confidence:.2f})")
                else:
                    with self.lock:
                        self.bird_name = "No target bird sound"
                        self.confidence = confidence
                    print(f"Audio Detection: Other sound ({self.confidence:.2f})")
            except Exception as e:
                print(f"Audio processing error: {e}")
                continue

    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def get_bird_name(self):
        with self.lock:
            return self.bird_name, self.confidence

def load_models():
    global vision_model, model
    print("Loading vision model...")
    vision_model = get_roboflow_model(
        model_id="birdvision-india/4",
        api_key=api_key
    )
    print("Vision model loaded successfully")
    print("Loading audio model...")
    model = load_model('model_16_val_loss_0.1970.keras')
    print("Audio model loaded successfully")

def main_detection():
    global video_running, model, vision_model
    try:
        if vision_model is None or model is None:
            load_models()
        cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)
        tracker = sv.ByteTrack()
        smoother = sv.DetectionsSmoother()
        bbox_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        audio_processor = AudioProcessor()
        audio_processor.start()
        while video_running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            results = vision_model.infer(frame, confidence=CONFIDENCE_THRESHOLD)[0]
            detections = sv.Detections.from_inference(results)
            high_conf_indices = [i for i, conf in enumerate(detections.confidence) if conf > CONFIDENCE_THRESHOLD]
            detections = detections[high_conf_indices]
            detections = tracker.update_with_detections(detections)
            detections = smoother.update_with_detections(detections)
            class_names = []
            if len(detections) > 0:
                for i in range(len(detections)):
                    try:
                        visual_pred = results.predictions[high_conf_indices[i]]
                        if visual_pred.confidence > CONFIDENCE_THRESHOLD:
                            visual_info = f"{visual_pred.class_name} {visual_pred.confidence:.2f}"
                        else:
                            visual_info = f"Unconfirmed bird {visual_pred.confidence:.2f}"
                        bird_name, confidence = audio_processor.get_bird_name()
                        audio_info = ""
                        if bird_name is not None:
                            if confidence > 0.85:
                                audio_info = f" | Audio: {bird_name} ({confidence:.2f})"
                            else:
                                audio_info = f" | Audio: No target bird sound ({confidence:.2f})"
                        class_names.append(f"{visual_info}{audio_info}")
                    except IndexError:
                        continue
            annotated_frame = bbox_annotator.annotate(scene=frame.copy(), detections=detections)
            if len(class_names) > 0:
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections[:len(class_names)],
                    labels=class_names
                )
            bird_name, confidence = audio_processor.get_bird_name()
            if len(detections) == 0 and bird_name is not None:
                if confidence > 0.85:
                    text = f"Audio Detection: {bird_name} ({confidence:.2f})"
                else:
                    text = f"Audio Detection: No target bird sound ({confidence:.2f})"
                cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("BirdVision India Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        audio_processor.stop()
        audio_processor.join()
        cap.release()
        cv2.destroyAllWindows()

def predict_from_audio(file_path):
    try:
        print(f"Loading audio file: {file_path}")
        waveform, sr = librosa.load(file_path, sr=None)
        print(f"Audio loaded successfully. Sample rate: {sr}, Waveform shape: {waveform.shape}")
        waveform_trimmed, _ = librosa.effects.trim(waveform, top_db=20)
        total_duration = librosa.get_duration(y=waveform_trimmed, sr=sr)
        num_chunks = max(1, int(total_duration // 5))
        print(f"Audio duration: {total_duration}s, Number of chunks: {num_chunks}")
        class_predictions = np.zeros(model.output_shape[1])
        chunk_predictions = []
        mel_specs_images = []
        for i in range(num_chunks):
            print(f"Processing chunk {i+1}/{num_chunks}")
            start_sample = int(i * 5 * sr)
            end_sample = start_sample + int(5.0 * sr)
            waveform_5sec = waveform_trimmed[start_sample:end_sample]
            print(f"Creating mel spectrogram for chunk {i+1}")
            mel_spectrogram = librosa.feature.melspectrogram(
                y=waveform_5sec, sr=sr, n_mels=224, hop_length=512
            )
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            print(f"Converting spectrogram to image for chunk {i+1}")
            buf = io.BytesIO()
            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            ax.axis('off')
            librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512, cmap='magma')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            mel_specs_images.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            print(f"Preparing model input for chunk {i+1}")
            img = Image.open(buf).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            print(f"Making prediction for chunk {i+1}")
            prediction = model.predict(img_array, verbose=0)
            chunk_predictions.append(prediction[0])
            class_predictions += prediction[0]
            print(f"Chunk {i+1} predictions: {prediction[0]}")
        print("Calculating final predictions")
        avg_predictions = class_predictions / num_chunks
        max_confidence = np.max(avg_predictions)
        predicted_class = np.argmax(avg_predictions)
        sorted_indices = np.argsort(avg_predictions)[::-1]
        top_predictions = [(label_map[idx], float(avg_predictions[idx])) for idx in sorted_indices[:3]]
        if max_confidence >= 0.90:
            predicted_label = label_map[predicted_class]
            status = "confident_match"
        elif max_confidence >= 0.80:
            predicted_label = f"Close to {label_map[predicted_class]}"
            status = "close_match"
        else:
            predicted_label = "No target bird call detected"
            status = "uncertain_match"
        print(f"Final prediction: {predicted_label} with confidence {max_confidence}")
        return predicted_class, predicted_label, avg_predictions, max_confidence, mel_specs_images, status, top_predictions
    except Exception as e:
        print(f"Error in predict_from_audio: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg')):
        return jsonify({'error': 'Invalid file format'}), 400
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(temp_path)
    try:
        predicted_class, predicted_label, predictions, confidence, mel_specs, status, top_predictions = predict_from_audio(temp_path)
        result = {
            'success': True,
            'status': status,
            'predicted_class': int(predicted_class),
            'predicted_label': predicted_label,
            'confidence': float(confidence),
            'mel_spectrograms': mel_specs,
            'predictions': {label_map[i]: float(pred) for i, pred in enumerate(predictions)},
            'top_matches': [{'label': label, 'confidence': conf} for label, conf in top_predictions]
        }
        return jsonify(result)
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Home page
@app.route('/')
def home():
    # Generate the map visualization
    metadata_df = pd.read_csv('train_metadata - Copy.csv')
    
    # Define approximate bounding box for India
    india_df = metadata_df[
        (metadata_df['latitude'] >= 6) & (metadata_df['latitude'] <= 38) &
        (metadata_df['longitude'] >= 68) & (metadata_df['longitude'] <= 98)
    ]
    
    # Create scatter plot for India only
    fig = px.scatter_map(india_df, lat='latitude', lon='longitude', color='common_name', 
                        hover_name='common_name', hover_data=['latitude', 'longitude'], 
                        title='Geographical Distribution of Bird Species in India',
                        height=600)
    fig.update_layout(mapbox_style="open-street-map")
    
    # Convert the plot to HTML
    map_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    return render_template('home.html', map_html=map_html)

# Audio detection page
@app.route('/audio')
def audio():
    return render_template('audio.html')

# Image/Video detection page
@app.route('/image-video')
def image_video():
    return render_template('image_video.html')

# Real-time detection page
@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/start_video')
def start_video():
    global video_running
    if not video_running:
        video_running = True
        thread = threading.Thread(target=main_detection)
        thread.start()
    return jsonify({"status": "success"})

@app.route('/stop_video')
def stop_video():
    global video_running
    video_running = False
    return jsonify({"status": "success"})

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'error': 'File type not allowed'}), 400
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        frame = cv2.imread(temp_path)
        results = vision_model.infer(frame, confidence=0.78)[0]
        detections = sv.Detections.from_inference(results)
        high_conf_indices = [i for i, conf in enumerate(detections.confidence) if conf > 0.78]
        detections = detections[high_conf_indices]
        bbox_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        class_names = []
        for i in range(len(detections)):
            try:
                visual_pred = results.predictions[high_conf_indices[i]]
                visual_info = f"{visual_pred.class_name} {visual_pred.confidence:.2f}"
                class_names.append(visual_info)
            except IndexError:
                continue
        annotated_frame = bbox_annotator.annotate(scene=frame.copy(), detections=detections)
        if len(class_names) > 0:
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections[:len(class_names)],
                labels=class_names
            )
        output_filename = f'result_{filename}'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, annotated_frame)
        os.remove(temp_path)
        return jsonify({
            'success': True,
            'result_path': f'/static/uploads/{output_filename}',
            'detections': class_names
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'error': 'File type not allowed'}), 400
    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(video_path)
        cap = cv2.VideoCapture(video_path)
        tracker = sv.ByteTrack()
        smoother = sv.DetectionsSmoother()
        bbox_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = vision_model.infer(frame, confidence=0.78)[0]
            detections = sv.Detections.from_inference(results)
            high_conf_indices = [i for i, conf in enumerate(detections.confidence) if conf > 0.78]
            detections = detections[high_conf_indices]
            detections = tracker.update_with_detections(detections)
            detections = smoother.update_with_detections(detections)
            class_names = []
            for i in range(len(detections)):
                try:
                    visual_pred = results.predictions[high_conf_indices[i]]
                    visual_info = f"{visual_pred.class_name} {visual_pred.confidence:.2f}"
                    class_names.append(visual_info)
                except IndexError:
                    continue
            annotated_frame = bbox_annotator.annotate(scene=frame.copy(), detections=detections)
            if len(class_names) > 0:
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections[:len(class_names)],
                    labels=class_names
                )
            cv2.imshow("Bird Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        os.remove(video_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True) 