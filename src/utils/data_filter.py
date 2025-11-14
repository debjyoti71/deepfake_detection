import pandas as pd
import os
import shutil
from src.config import Config

class DataFilter:
    def __init__(self, metadata_path='data/raw/meta_data.csv'):
        self.metadata_path = metadata_path
        self.df = pd.read_csv(metadata_path)
        
    def filter_data(self):
        """Filter dataset based on configuration"""
        filtered_df = self.df.copy()
        
        # Filter by race
        if Config.FILTER_BY_RACE and Config.SELECTED_RACES:
            filtered_df = filtered_df[filtered_df['race'].isin(Config.SELECTED_RACES)]
            print(f"Filtered by race: {Config.SELECTED_RACES}")
        
        # Filter by gender
        if Config.FILTER_BY_GENDER and Config.SELECTED_GENDERS:
            filtered_df = filtered_df[filtered_df['gender'].isin(Config.SELECTED_GENDERS)]
            print(f"Filtered by gender: {Config.SELECTED_GENDERS}")
        
        # Filter by category
        if Config.FILTER_BY_CATEGORY and Config.SELECTED_CATEGORIES:
            filtered_df = filtered_df[filtered_df['category'].isin(Config.SELECTED_CATEGORIES)]
            print(f"Filtered by category: {Config.SELECTED_CATEGORIES}")
        
        # Limit samples per category
        if Config.MAX_SAMPLES_PER_CATEGORY > 0:
            filtered_df = filtered_df.groupby('category').head(Config.MAX_SAMPLES_PER_CATEGORY)
            print(f"Limited to {Config.MAX_SAMPLES_PER_CATEGORY} samples per category")
        
        # Limit total samples
        if Config.MAX_TOTAL_SAMPLES > 0 and len(filtered_df) > Config.MAX_TOTAL_SAMPLES:
            filtered_df = filtered_df.sample(n=Config.MAX_TOTAL_SAMPLES, random_state=42)
            print(f"Limited to {Config.MAX_TOTAL_SAMPLES} total samples")
        
        print(f"Final dataset size: {len(filtered_df)} samples")
        return filtered_df
    
    def get_file_paths(self, filtered_df):
        """Get actual file paths for filtered data"""
        file_paths = []
        for _, row in filtered_df.iterrows():
            video_path = os.path.join('data/raw', row['full_path'], row['path'])
            if os.path.exists(video_path):
                file_paths.append({
                    'video_path': video_path,
                    'category': row['category'],
                    'race': row['race'],
                    'gender': row['gender'],
                    'method': row['method']
                })
        return file_paths
    
    def clean_processed_data(self):
        """Remove processed data that doesn't match current filter"""
        print("Cleaning processed data directories...")
        
        # Clean directories
        dirs_to_clean = [
            Config.PROCESSED_DATA_DIR + '/frames',
            Config.PROCESSED_DATA_DIR + '/faces', 
            Config.PROCESSED_DATA_DIR + '/audio',
            Config.PROCESSED_DATA_DIR + '/audio_features'
        ]
        
        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                print(f"Cleaned {dir_path}")