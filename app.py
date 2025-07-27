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
import logging
from config import Config

matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Create upload directory
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# --- CONFIG ---
USE_WEBCAM = True
VIDEO_PATH = "BK.mp4"
AUDIO_FORMAT = pyaudio.paFloat32
AUDIO_CHUNK = Config.AUDIO_RATE * Config.AUDIO_CHUNK_DURATION

# Global variables
video_running = False
audio_model = None
vision_model = None

def load_models():
    """Load AI models with proper error handling"""
    global vision_model, audio_model
    try:
        logger.info("Loading audio model...")
        audio_model = load_model(Config.AUDIO_MODEL_PATH)
        logger.info("Audio model loaded successfully")
        
        logger.info("Loading vision model...")
        if Config.ROBOFLOW_API_KEY == 'your-roboflow-api-key-here':
            logger.warning("Using placeholder API key. Please set ROBOFLOW_API_KEY environment variable.")
        vision_model = get_roboflow_model(
            model_id=Config.ROBOFLOW_MODEL_ID,
            api_key=Config.ROBOFLOW_API_KEY
        )
        logger.info("Vision model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

class AudioProcessor(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=AUDIO_FORMAT,
            channels=Config.AUDIO_CHANNELS,
            rate=Config.AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK
        )
        self.running = True
        self.latest_prediction = None
        self.bird_name = None
        self.confidence = 0.0
        self.lock = threading.Lock()

    def process_audio_chunk(self, audio_data):
        """Process audio data and convert to mel-spectrogram image"""
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=Config.AUDIO_RATE,
            n_mels=224,
            hop_length=512
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.figure(figsize=(2.24, 2.24), dpi=100)
            plt.axis('off')
            librosa.display.specshow(mel_spectrogram_db, sr=Config.AUDIO_RATE, hop_length=512, cmap='magma')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(tmpfile.name, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            img = Image.open(tmpfile.name).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Clean up temp file
            os.unlink(tmpfile.name)
            
        return img_array

    def run(self):
        global audio_model
        while self.running:
            try:
                audio_data = np.frombuffer(
                    self.stream.read(AUDIO_CHUNK, exception_on_overflow=False),
                    dtype=np.float32
                )
                img_array = self.process_audio_chunk(audio_data)
                prediction = audio_model.predict(img_array, verbose=0)
                self.latest_prediction = prediction
                predicted_class = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class])
                
                if confidence > Config.AUDIO_CONFIDENCE_THRESHOLD:
                    with self.lock:
                        self.bird_name = Config.BIRD_CLASSES.get(predicted_class, "Unknown")
                        self.confidence = confidence
                    logger.info(f"Audio Detection: {self.bird_name} ({self.confidence:.2f})")
                else:
                    with self.lock:
                        self.bird_name = "No target bird sound"
                        self.confidence = confidence
                    logger.info(f"Audio Detection: Other sound ({self.confidence:.2f})")
                    
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                continue

    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def get_bird_name(self):
        with self.lock:
            return self.bird_name, self.confidence

def main_detection():
    """Main detection function for real-time video processing"""
    global video_running, audio_model, vision_model
    try:
        if vision_model is None or audio_model is None:
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
                logger.error("Failed to capture frame")
                break
                
            results = vision_model.infer(frame, confidence=Config.CONFIDENCE_THRESHOLD)[0]
            detections = sv.Detections.from_inference(results)
            high_conf_indices = [i for i, conf in enumerate(detections.confidence) 
                               if conf > Config.CONFIDENCE_THRESHOLD]
            detections = detections[high_conf_indices]
            detections = tracker.update_with_detections(detections)
            detections = smoother.update_with_detections(detections)
            
            class_names = []
            if len(detections) > 0:
                for i in range(len(detections)):
                    try:
                        visual_pred = results.predictions[high_conf_indices[i]]
                        if visual_pred.confidence > Config.CONFIDENCE_THRESHOLD:
                            visual_info = f"{visual_pred.class_name} {visual_pred.confidence:.2f}"
                        else:
                            visual_info = f"Unconfirmed bird {visual_pred.confidence:.2f}"
                        
                        bird_name, confidence = audio_processor.get_bird_name()
                        audio_info = ""
                        if bird_name is not None:
                            if confidence > Config.AUDIO_CONFIDENCE_THRESHOLD:
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
                if confidence > Config.AUDIO_CONFIDENCE_THRESHOLD:
                    text = f"Audio Detection: {bird_name} ({confidence:.2f})"
                else:
                    text = f"Audio Detection: No target bird sound ({confidence:.2f})"
                cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.imshow("BirdVision India Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logger.error(f"Error in main detection: {e}")
    finally:
        if 'audio_processor' in locals():
            audio_processor.stop()
            audio_processor.join()
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

def predict_from_audio(file_path):
    """Predict bird species from audio file with improved error handling"""
    global audio_model
    try:
        if audio_model is None:
            load_models()
            
        logger.info(f"Loading audio file: {file_path}")
        waveform, sr = librosa.load(file_path, sr=None)
        logger.info(f"Audio loaded successfully. Sample rate: {sr}, Waveform shape: {waveform.shape}")
        
        waveform_trimmed, _ = librosa.effects.trim(waveform, top_db=20)
        total_duration = librosa.get_duration(y=waveform_trimmed, sr=sr)
        num_chunks = max(1, int(total_duration // Config.AUDIO_CHUNK_DURATION))
        logger.info(f"Audio duration: {total_duration}s, Number of chunks: {num_chunks}")
        
        class_predictions = np.zeros(audio_model.output_shape[1])
        chunk_predictions = []
        mel_specs_images = []
        
        for i in range(num_chunks):
            logger.info(f"Processing chunk {i+1}/{num_chunks}")
            start_sample = int(i * Config.AUDIO_CHUNK_DURATION * sr)
            end_sample = start_sample + int(Config.AUDIO_CHUNK_DURATION * sr)
            waveform_5sec = waveform_trimmed[start_sample:end_sample]
            
            # Create mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=waveform_5sec, sr=sr, n_mels=224, hop_length=512
            )
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Convert to image
            buf = io.BytesIO()
            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            ax.axis('off')
            librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512, cmap='magma')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            mel_specs_images.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            
            # Prepare model input
            img = Image.open(buf).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = audio_model.predict(img_array, verbose=0)
            chunk_predictions.append(prediction[0])
            class_predictions += prediction[0]
            
        # Calculate final predictions
        avg_predictions = class_predictions / num_chunks
        max_confidence = np.max(avg_predictions)
        predicted_class = np.argmax(avg_predictions)
        sorted_indices = np.argsort(avg_predictions)[::-1]
        top_predictions = [(Config.BIRD_CLASSES[idx], float(avg_predictions[idx])) 
                          for idx in sorted_indices[:3]]
        
        if max_confidence >= 0.90:
            predicted_label = Config.BIRD_CLASSES[predicted_class]
            status = "confident_match"
        elif max_confidence >= 0.80:
            predicted_label = f"Close to {Config.BIRD_CLASSES[predicted_class]}"
            status = "close_match"
        else:
            predicted_label = "No target bird call detected"
            status = "uncertain_match"
            
        logger.info(f"Final prediction: {predicted_label} with confidence {max_confidence}")
        return predicted_class, predicted_label, avg_predictions, max_confidence, mel_specs_images, status, top_predictions
        
    except Exception as e:
        logger.error(f"Error in predict_from_audio: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """Audio prediction endpoint with improved validation"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not allowed_file(file.filename, Config.ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({'error': 'Invalid file format. Allowed: ' + ', '.join(Config.ALLOWED_AUDIO_EXTENSIONS)}), 400
    
    # Secure filename and save
    filename = secure_filename(file.filename)
    temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    try:
        file.save(temp_path)
        predicted_class, predicted_label, predictions, confidence, mel_specs, status, top_predictions = predict_from_audio(temp_path)
        
        result = {
            'success': True,
            'status': status,
            'predicted_class': int(predicted_class),
            'predicted_label': predicted_label,
            'confidence': float(confidence),
            'mel_spectrograms': mel_specs,
            'predictions': {Config.BIRD_CLASSES[i]: float(pred) for i, pred in enumerate(predictions)},
            'top_matches': [{'label': label, 'confidence': conf} for label, conf in top_predictions]
        }
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Ensure temp file is cleaned up
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_path}: {e}")

# Home page
@app.route('/')
def home():
    """Home page with improved error handling for map visualization"""
    try:
        # Generate the map visualization
        metadata_df = pd.read_csv(Config.METADATA_CSV_PATH)
        
        # Define approximate bounding box for India
        lat_min, lat_max = Config.INDIA_LAT_BOUNDS
        lon_min, lon_max = Config.INDIA_LON_BOUNDS
        
        india_df = metadata_df[
            (metadata_df['latitude'] >= lat_min) & (metadata_df['latitude'] <= lat_max) &
            (metadata_df['longitude'] >= lon_min) & (metadata_df['longitude'] <= lon_max)
        ]
        
        # Create scatter plot for India only
        color_column = 'common_name' if 'common_name' in india_df.columns else 'primary_label'
        fig = px.scatter_map(india_df, lat='latitude', lon='longitude', color=color_column, 
                            hover_name=color_column, hover_data=['latitude', 'longitude'], 
                            title='Geographical Distribution of Bird Species in India',
                            height=600)
        fig.update_layout(mapbox_style="open-street-map")
        
        # Convert the plot to HTML
        map_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"Error generating map: {e}")
        map_html = f"<p class='text-danger'>Error loading map: {str(e)}</p>"
    
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

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if models are loaded
        models_status = {
            'audio_model': audio_model is not None,
            'vision_model': vision_model is not None
        }
        
        # Check upload directory
        upload_dir_exists = os.path.exists(Config.UPLOAD_FOLDER)
        
        # Check required files
        required_files = {
            'audio_model_file': os.path.exists(Config.AUDIO_MODEL_PATH),
            'metadata_file': os.path.exists(Config.METADATA_CSV_PATH)
        }
        
        all_healthy = (
            all(models_status.values()) and 
            upload_dir_exists and 
            all(required_files.values())
        )
        
        status_code = 200 if all_healthy else 503
        
        return jsonify({
            'status': 'healthy' if all_healthy else 'unhealthy',
            'models': models_status,
            'upload_directory': upload_dir_exists,
            'required_files': required_files,
            'timestamp': pd.Timestamp.now().isoformat()
        }), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': pd.Timestamp.now().isoformat()
        }), 503

@app.route('/start_video')
def start_video():
    """Start real-time video detection"""
    global video_running
    
    try:
        if video_running:
            return jsonify({"status": "already_running", "message": "Video detection is already running"})
        
        # Ensure models are loaded
        if vision_model is None or audio_model is None:
            load_models()
        
        video_running = True
        thread = threading.Thread(target=main_detection, daemon=True)
        thread.start()
        
        logger.info("Video detection started")
        return jsonify({"status": "success", "message": "Video detection started"})
        
    except Exception as e:
        logger.error(f"Failed to start video detection: {e}")
        video_running = False
        return jsonify({"status": "error", "message": f"Failed to start: {str(e)}"})

@app.route('/stop_video')
def stop_video():
    """Stop real-time video detection"""
    global video_running
    
    try:
        if not video_running:
            return jsonify({"status": "already_stopped", "message": "Video detection is not running"})
        
        video_running = False
        logger.info("Video detection stopped")
        return jsonify({"status": "success", "message": "Video detection stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping video detection: {e}")
        return jsonify({"status": "error", "message": f"Error stopping: {str(e)}"})

@app.route('/video_status')
def video_status():
    """Get current video detection status"""
    return jsonify({
        "running": video_running,
        "models_loaded": {
            "audio": audio_model is not None,
            "vision": vision_model is not None
        }
    })

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process uploaded image for bird detection"""
    global vision_model
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename, Config.ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'error': 'File type not allowed. Allowed: ' + ', '.join(Config.ALLOWED_IMAGE_EXTENSIONS)}), 400
    
    filename = secure_filename(file.filename)
    temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    try:
        if vision_model is None:
            load_models()
            
        file.save(temp_path)
        frame = cv2.imread(temp_path)
        
        if frame is None:
            return jsonify({'error': 'Invalid image file'}), 400
            
        results = vision_model.infer(frame, confidence=Config.CONFIDENCE_THRESHOLD)[0]
        detections = sv.Detections.from_inference(results)
        high_conf_indices = [i for i, conf in enumerate(detections.confidence) 
                           if conf > Config.CONFIDENCE_THRESHOLD]
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
        output_path = os.path.join(Config.UPLOAD_FOLDER, output_filename)
        cv2.imwrite(output_path, annotated_frame)
        
        return jsonify({
            'success': True,
            'result_path': f'/static/uploads/{output_filename}',
            'detections': class_names
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_path}: {e}")

@app.route('/process_video', methods=['POST'])
def process_video():
    """Process uploaded video for bird detection"""
    global vision_model
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename, Config.ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'error': 'File type not allowed. Allowed: ' + ', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}), 400
    
    filename = secure_filename(file.filename)
    video_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    try:
        if vision_model is None:
            load_models()
            
        file.save(video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return jsonify({'error': 'Invalid video file'}), 400
            
        tracker = sv.ByteTrack()
        smoother = sv.DetectionsSmoother()
        bbox_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        frame_count = 0
        detections_found = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            results = vision_model.infer(frame, confidence=Config.CONFIDENCE_THRESHOLD)[0]
            detections = sv.Detections.from_inference(results)
            high_conf_indices = [i for i, conf in enumerate(detections.confidence) 
                               if conf > Config.CONFIDENCE_THRESHOLD]
            detections = detections[high_conf_indices]
            detections = tracker.update_with_detections(detections)
            detections = smoother.update_with_detections(detections)
            
            class_names = []
            for i in range(len(detections)):
                try:
                    visual_pred = results.predictions[high_conf_indices[i]]
                    visual_info = f"{visual_pred.class_name} {visual_pred.confidence:.2f}"
                    class_names.append(visual_info)
                    detections_found.append({
                        'frame': frame_count,
                        'species': visual_pred.class_name,
                        'confidence': visual_pred.confidence
                    })
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
        
        return jsonify({
            'success': True,
            'total_frames': frame_count,
            'detections': detections_found,
            'message': f'Processed {frame_count} frames with {len(detections_found)} detections'
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                logger.warning(f"Could not remove temp file {video_path}: {e}")

@app.route('/api/docs')
def api_docs():
    """Simple API documentation"""
    docs = {
        "name": "BirdVision India API",
        "version": "1.0.0",
        "description": "Bird detection API using audio and visual inputs",
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "Home page with species map",
                "returns": "HTML page"
            },
            "/health": {
                "method": "GET", 
                "description": "Health check endpoint",
                "returns": "JSON with system status"
            },
            "/predict": {
                "method": "POST",
                "description": "Audio bird detection",
                "parameters": {
                    "audio": "Audio file (wav, mp3, ogg, flac)"
                },
                "returns": "JSON with predictions and confidence scores"
            },
            "/process_image": {
                "method": "POST",
                "description": "Image bird detection",
                "parameters": {
                    "file": "Image file (png, jpg, jpeg, gif)"
                },
                "returns": "JSON with detection results and annotated image path"
            },
            "/process_video": {
                "method": "POST",
                "description": "Video bird detection",
                "parameters": {
                    "file": "Video file (mp4, avi, mov)"
                },
                "returns": "JSON with detection summary"
            },
            "/start_video": {
                "method": "GET",
                "description": "Start real-time detection",
                "returns": "JSON status"
            },
            "/stop_video": {
                "method": "GET",
                "description": "Stop real-time detection", 
                "returns": "JSON status"
            },
            "/video_status": {
                "method": "GET",
                "description": "Get real-time detection status",
                "returns": "JSON with current status"
            }
        },
        "bird_species": Config.BIRD_CLASSES,
        "supported_formats": {
            "audio": list(Config.ALLOWED_AUDIO_EXTENSIONS),
            "image": list(Config.ALLOWED_IMAGE_EXTENSIONS),
            "video": list(Config.ALLOWED_VIDEO_EXTENSIONS)
        }
    }
    return jsonify(docs)

if __name__ == '__main__':
    # Initialize models on startup in production
    if not app.debug:
        try:
            load_models()
        except Exception as e:
            logger.error(f"Failed to load models on startup: {e}")
    
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
    # Initialize models on startup in production
    if not app.debug:
        try:
            load_models()
        except Exception as e:
            logger.error(f"Failed to load models on startup: {e}")
    
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
