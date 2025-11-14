#!/usr/bin/env python3
"""
Main pipeline runner for deepfake detection system with balanced sampling
"""

import os
import sys
from dotenv import load_dotenv
from src.preprocessing.data_processor import DataProcessor
from src.utils.logger import setup_logger

def main():
    # Setup logger
    logger = setup_logger('Pipeline')
    logger.info("=== Deepfake Detection Pipeline (Balanced Sampling) ===")
    
    # Load environment variables
    load_dotenv()
    
    # Display configuration
    print(f"Configuration:")
    print(f"  - Use balanced sampling: {os.getenv('USE_BALANCED_SAMPLING', 'True')}")
    print(f"  - Filter by race: {os.getenv('FILTER_BY_RACE', 'False')}")
    if os.getenv('FILTER_BY_RACE', 'False').lower() == 'true':
        print(f"  - Selected races: {os.getenv('SELECTED_RACES', '')}")
    print(f"  - Max samples per real: {os.getenv('MAX_SAMPLES_PER_REAL', '25')}")
    print(f"  - Batch size: {os.getenv('BATCH_SIZE', '16')}")
    print()
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        # Step 1: Data Processing with Balanced Sampling
        print("Step 1: Processing data with balanced sampling...")
        processor = DataProcessor()
        df, copied_files, missing_files = processor.process_data()
        
        print(f"[SUCCESS] Data processing completed:")
        print(f"  - Total samples: {len(df)}")
        print(f"  - Files copied: {len(copied_files)}")
        print(f"  - Missing files: {len(missing_files)}")
        print(f"  - Category distribution: {df['category'].value_counts().to_dict()}")
        
        # Step 2: Model Training
        print("\nStep 2: Training models...")
        from src.training.trainer import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_all_models()
        
        # Step 3: Model Evaluation
        print("\nStep 3: Evaluating models...")
        from src.evaluation.evaluator import ModelEvaluator
        evaluator = ModelEvaluator()
        evaluator.evaluate_all_models()
        
        print("\n[SUCCESS] Pipeline setup completed successfully!")
        print(f"[SUCCESS] Balanced dataset created with {len(df)} samples")
        print(f"[SUCCESS] Results will be saved to: results/")
        print(f"[SUCCESS] Models will be saved to: models/")
        
        print("\n[SUCCESS] Complete pipeline executed!")
        print("Results available in:")
        print("  - Models: models/")
        print("  - Evaluation: results/")
        print("  - Confusion matrices: results/*_confusion_matrix.png")
        print("  - Detailed report: results/evaluation_report.txt")
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()