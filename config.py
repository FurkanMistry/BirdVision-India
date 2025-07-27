import os

class Config:
    """Configuration settings for BirdVision India application"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'birdvision-india-secret-key-2024'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Roboflow API Configuration
    ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY') or 'your-roboflow-api-key-here'
    ROBOFLOW_MODEL_ID = "birdvision-india/4"
    
    # Detection Configuration
    CONFIDENCE_THRESHOLD = 0.78
    AUDIO_CONFIDENCE_THRESHOLD = 0.85
    
    # Audio Configuration
    AUDIO_RATE = 22050
    AUDIO_CHANNELS = 1
    AUDIO_CHUNK_DURATION = 5
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
    ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
    
    # Model Paths
    AUDIO_MODEL_PATH = 'model_16_val_loss_0.1970.keras'
    METADATA_CSV_PATH = 'train_metadata - Copy.csv'
    
    # Bird Species Configuration
    BIRD_CLASSES = {
        0: 'Asian Koel',
        1: 'Black Kite',
        2: 'Common Kingfisher',
        3: 'Common Myna',
        4: 'House Sparrow',
        5: 'Little Ringed Plover',
        6: 'Rose Ringed Parakeet',  # Fixed typo from 'Paraket'
        7: 'Whiskered Tern'
    }
    
    # India Geographic Bounds for Map
    INDIA_LAT_BOUNDS = (6, 38)
    INDIA_LON_BOUNDS = (68, 98)