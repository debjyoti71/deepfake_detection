#!/usr/bin/env python3
"""
Test the Deepfake Detection API
"""

import requests
import json
from pathlib import Path

def test_api():
    """Test API with a sample video"""
    
    # API endpoint
    url = "http://localhost:8000/predict"
    
    # Find a test video from processed data
    test_video = Path("data/raw/FakeVideo-FakeAudio/African/men/id00076/00109_2_id00166_wavtolip.mp4")
    
    if not test_video.exists():
        # Try processed data instead
        test_video_dir = Path("data/processed/videos/A/Asian (East)/men")
        test_videos = list(test_video_dir.glob("*.mp4"))
        if test_videos:
            test_video = test_videos[0]
        else:
            print("No test videos found. Run the pipeline first.")
            return
    
    print(f"Testing with video: {test_video}")
    

    
    try:
        # Upload video to API
        with open(test_video, 'rb') as f:
            files = {'file': (test_video.name, f, 'video/mp4')}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== API Response ===")
            print(f"Filename: {result['filename']}")
            print(f"File size: {result['file_size']} bytes")
            print(f"Overall Prediction: {result['overall_prediction']}")
            print(f"Prediction Time: {result['prediction_time_seconds']}s")
            print("\n--- Predictions ---")
            
            for model_type in ['visual', 'audio', 'fusion']:
                pred = result['predictions'][model_type]
                print(f"{model_type.upper()}: {pred['prediction']} (confidence: {pred['confidence']:.3f})")
                
            print("\n--- Detailed Scores ---")
            fusion_scores = result['predictions']['fusion']['scores']
            for category, score in fusion_scores.items():
                category_name = result['categories'][category]
                print(f"{category}: {score:.3f} - {category_name}")
                
        else:
            print(f"API Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running.")
        print("Start the server with: python start_api.py")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()