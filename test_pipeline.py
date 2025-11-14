#!/usr/bin/env python3
"""
Test script to verify the pipeline works
"""

import os
from src.config import Config
from src.utils.data_filter import DataFilter

def test_data_filtering():
    """Test data filtering functionality"""
    print("=== Testing Data Filtering ===")
    
    filter = DataFilter()
    filtered_df = filter.filter_data()
    
    print(f"Filtered dataset contains {len(filtered_df)} samples")
    print(f"Categories: {filtered_df['category'].value_counts().to_dict()}")
    print(f"Races: {filtered_df['race'].value_counts().to_dict()}")
    print()

def test_file_paths():
    """Test if processed files exist"""
    print("=== Testing File Paths ===")
    
    filter = DataFilter()
    filtered_df = filter.filter_data()
    file_paths = filter.get_file_paths(filtered_df)
    
    print(f"Found {len(file_paths)} valid file paths")
    
    # Check if processed files exist
    faces_exist = 0
    audio_exist = 0
    
    for item in file_paths[:5]:  # Check first 5
        base_name = os.path.splitext(os.path.basename(item['video_path']))[0]
        face_path = os.path.join(Config.PROCESSED_DATA_DIR, 'faces', f"{base_name}_face_0000.jpg")
        audio_path = os.path.join(Config.PROCESSED_DATA_DIR, 'audio', f"{base_name}.wav")
        
        if os.path.exists(face_path):
            faces_exist += 1
        if os.path.exists(audio_path):
            audio_exist += 1
    
    print(f"Processed faces found: {faces_exist}/5")
    print(f"Processed audio found: {audio_exist}/5")
    print()

def main():
    print("=== Pipeline Test ===")
    print(f"Configuration:")
    print(f"  - Filter by race: {Config.FILTER_BY_RACE}")
    print(f"  - Selected races: {Config.SELECTED_RACES}")
    print(f"  - Max samples per category: {Config.MAX_SAMPLES_PER_CATEGORY}")
    print()
    
    test_data_filtering()
    test_file_paths()
    
    print("Test completed!")

if __name__ == "__main__":
    main()