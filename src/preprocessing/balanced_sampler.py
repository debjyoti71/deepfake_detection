import pandas as pd
import os
from collections import defaultdict
import random
from src.utils.logger import setup_logger

class BalancedSampler:
    def __init__(self, metadata_path, max_samples_per_real=25):
        self.metadata_path = metadata_path
        self.max_samples_per_real = max_samples_per_real
        self.df = pd.read_csv(metadata_path)
        self.logger = setup_logger('BalancedSampler')
        
    def create_balanced_sample(self, race_filter=None, gender_filter=None):
        """
        Create balanced sample where for each real-real video, 
        we get corresponding fake versions (B, C, D categories)
        """
        # Filter by race and gender if specified
        filtered_df = self.df.copy()
        
        if race_filter:
            if isinstance(race_filter, str):
                race_filter = [race_filter]
            filtered_df = filtered_df[filtered_df['race'].isin(race_filter)]
        if gender_filter:
            if isinstance(gender_filter, str):
                gender_filter = [gender_filter]
            filtered_df = filtered_df[filtered_df['gender'].isin(gender_filter)]
            
        # Group by source ID to find corresponding videos
        source_groups = defaultdict(list)
        
        for _, row in filtered_df.iterrows():
            source_id = row['source']
            source_groups[source_id].append(row)
        
        balanced_samples = []
        
        # Get real-real videos first (category A)
        real_real_videos = filtered_df[filtered_df['category'] == 'A'].copy()
        
        # Limit real-real videos to max_samples_per_real
        if len(real_real_videos) > self.max_samples_per_real:
            real_real_videos = real_real_videos.sample(n=self.max_samples_per_real, random_state=42)
        
        self.logger.info(f"Selected {len(real_real_videos)} real-real videos")
        
        # For each real-real video, find corresponding fake versions
        for _, real_video in real_real_videos.iterrows():
            source_id = real_video['source']
            
            # Add the real-real video
            balanced_samples.append(real_video)
            
            # Find corresponding fake versions for this source
            source_videos = source_groups[source_id]
            
            # Get one sample from each fake category (B, C, D)
            categories_found = {'B': [], 'C': [], 'D': []}
            
            for video in source_videos:
                if video['category'] in categories_found:
                    categories_found[video['category']].append(video)
            
            # Add one sample from each fake category if available
            for category in ['B', 'C', 'D']:
                if categories_found[category]:
                    # Randomly select one if multiple available
                    selected = random.choice(categories_found[category])
                    balanced_samples.append(selected)
                    # print(f"  Found {category} for {source_id}")
                else:
                    print(f"  Missing {category} for {source_id}")
        
        balanced_df = pd.DataFrame(balanced_samples)
        
        # Log statistics
        self.logger.info("Balanced Sample Statistics:")
        self.logger.info(f"Total samples: {len(balanced_df)}")
        self.logger.info(f"Category distribution: {balanced_df['category'].value_counts().sort_index().to_dict()}")
        self.logger.info(f"Race distribution: {balanced_df['race'].value_counts().to_dict()}")
        self.logger.info(f"Gender distribution: {balanced_df['gender'].value_counts().to_dict()}")
        
        return balanced_df
    
    def save_balanced_sample(self, output_path, race_filter=None, gender_filter=None):
        """Save balanced sample to CSV"""
        balanced_df = self.create_balanced_sample(race_filter, gender_filter)
        balanced_df.to_csv(output_path, index=False)
        print(f"\nBalanced sample saved to: {output_path}")
        return balanced_df

if __name__ == "__main__":
    # Example usage
    metadata_path = "e:/my_projects/deepfake_detction_2/data/raw/meta_data.csv"
    sampler = BalancedSampler(metadata_path, max_samples_per_real=25)
    
    # Create balanced sample for Asian (East) race
    balanced_df = sampler.save_balanced_sample(
        output_path="e:/my_projects/deepfake_detction_2/data/balanced_sample.csv",
        race_filter="Asian (East)",
        gender_filter=None  # Include both men and women
    )