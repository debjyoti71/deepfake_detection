import os
import pandas as pd
import shutil
from pathlib import Path
from dotenv import load_dotenv
from .balanced_sampler import BalancedSampler

class DataProcessor:
    def __init__(self):
        load_dotenv()
        self.raw_data_dir = Path("data/raw")
        self.processed_data_dir = Path("data/processed")
        self.metadata_path = self.raw_data_dir / "meta_data.csv"
        
        # Configuration from .env
        self.filter_by_race = os.getenv('FILTER_BY_RACE', 'False').lower() == 'true'
        self.selected_races = os.getenv('SELECTED_RACES', '').split(',') if os.getenv('SELECTED_RACES') else []
        self.filter_by_gender = os.getenv('FILTER_BY_GENDER', 'False').lower() == 'true'
        self.selected_genders = os.getenv('SELECTED_GENDERS', '').split(',') if os.getenv('SELECTED_GENDERS') else []
        self.use_balanced_sampling = os.getenv('USE_BALANCED_SAMPLING', 'True').lower() == 'true'
        self.max_samples_per_real = int(os.getenv('MAX_SAMPLES_PER_REAL', '25'))
        
        # Create processed directories
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories for processed data"""
        dirs = [
            self.processed_data_dir,
            self.processed_data_dir / "frames",
            self.processed_data_dir / "faces", 
            self.processed_data_dir / "audio",
            self.processed_data_dir / "audio_features",
            self.processed_data_dir / "sync",
            self.processed_data_dir / "fusion_features"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def get_filtered_data(self):
        """Get filtered dataset based on configuration"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
        if self.use_balanced_sampling:
            print("Using balanced sampling strategy...")
            sampler = BalancedSampler(str(self.metadata_path), self.max_samples_per_real)
            
            race_filter = self.selected_races if self.filter_by_race and self.selected_races else None
            gender_filter = self.selected_genders[0] if self.filter_by_gender and len(self.selected_genders) == 1 else None
            
            balanced_df = sampler.create_balanced_sample(race_filter, gender_filter)
            
            # Save balanced sample for reference
            balanced_sample_path = self.processed_data_dir / "balanced_sample.csv"
            balanced_df.to_csv(balanced_sample_path, index=False)
            print(f"Balanced sample saved to: {balanced_sample_path}")
            
            return balanced_df
        else:
            # Original filtering logic
            df = pd.read_csv(self.metadata_path)
            
            if self.filter_by_race and self.selected_races:
                df = df[df['race'].isin(self.selected_races)]
                
            if self.filter_by_gender and self.selected_genders:
                df = df[df['gender'].isin(self.selected_genders)]
                
            return df
    
    def copy_selected_files(self, df):
        """Copy selected video files to processed directory"""
        copied_files = []
        missing_files = []
        
        for _, row in df.iterrows():
            source_path = Path(row['full_path'])
            video_file = row['path']
            
            # Construct full source path
            full_source_path = self.raw_data_dir / source_path.relative_to(Path("data/raw"))
            full_video_path = full_source_path / video_file
            
            if full_video_path.exists():
                # Create destination directory structure
                dest_dir = self.processed_data_dir / "videos" / row['category'] / row['race'] / row['gender']
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                dest_path = dest_dir / video_file
                if not dest_path.exists():
                    shutil.copy2(full_video_path, dest_path)
                    
                copied_files.append({
                    'source': str(full_video_path),
                    'destination': str(dest_path),
                    'category': row['category'],
                    'race': row['race'],
                    'gender': row['gender'],
                    'source_id': row['source']
                })
            else:
                missing_files.append(str(full_video_path))
                
        print(f"Copied {len(copied_files)} files")
        if missing_files:
            print(f"Missing {len(missing_files)} files")
            
        return copied_files, missing_files
    
    def process_data(self):
        """Main processing function"""
        print("Starting data processing...")
        
        # Get filtered data
        df = self.get_filtered_data()
        print(f"Selected {len(df)} samples for processing")
        
        # Copy files
        copied_files, missing_files = self.copy_selected_files(df)
        
        # Save processing summary
        summary = {
            'total_samples': len(df),
            'copied_files': len(copied_files),
            'missing_files': len(missing_files),
            'category_distribution': df['category'].value_counts().to_dict(),
            'race_distribution': df['race'].value_counts().to_dict(),
            'gender_distribution': df['gender'].value_counts().to_dict()
        }
        
        summary_path = self.processed_data_dir / "processing_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Data Processing Summary\n")
            f.write("=" * 30 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
                
        print(f"Processing summary saved to: {summary_path}")
        
        return df, copied_files, missing_files

if __name__ == "__main__":
    processor = DataProcessor()
    df, copied_files, missing_files = processor.process_data()