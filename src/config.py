import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Data Filtering
    FILTER_BY_RACE = os.getenv('FILTER_BY_RACE', 'False').lower() == 'true'
    SELECTED_RACES = os.getenv('SELECTED_RACES', '').split(',') if os.getenv('SELECTED_RACES') else []
    
    FILTER_BY_GENDER = os.getenv('FILTER_BY_GENDER', 'False').lower() == 'true'
    SELECTED_GENDERS = os.getenv('SELECTED_GENDERS', 'men,women').split(',')
    
    FILTER_BY_CATEGORY = os.getenv('FILTER_BY_CATEGORY', 'False').lower() == 'true'
    SELECTED_CATEGORIES = os.getenv('SELECTED_CATEGORIES', 'A,B,C,D').split(',')
    
    # Data Limits
    MAX_SAMPLES_PER_CATEGORY = int(os.getenv('MAX_SAMPLES_PER_CATEGORY', '100'))
    MAX_TOTAL_SAMPLES = int(os.getenv('MAX_TOTAL_SAMPLES', '1000'))
    
    # Thresholds
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
    DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.7'))

    
    # Processing
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    MAX_FRAMES_PER_VIDEO = int(os.getenv('MAX_FRAMES_PER_VIDEO', '16'))
    AUDIO_SAMPLE_RATE = int(os.getenv('AUDIO_SAMPLE_RATE', '16000'))
    IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', '224'))
    
    # Paths
    VISUAL_MODEL_PATH = os.getenv('VISUAL_MODEL_PATH', 'models/visual_model/visual_model.pth')
    AUDIO_MODEL_PATH = os.getenv('AUDIO_MODEL_PATH', 'models/audio_model/audio_model.pth')

    FUSION_MODEL_PATH = os.getenv('FUSION_MODEL_PATH', 'models/fusion_model/fusion_model.pth')
    
    PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR', 'data/processed')
    RESULTS_DIR = os.getenv('RESULTS_DIR', 'results')
    LOGS_DIR = os.getenv('LOGS_DIR', 'logs')