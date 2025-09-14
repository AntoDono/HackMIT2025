#!/usr/bin/env python3
"""
Brain Wave Training Script
Trains a CNN model on brainwave data to classify emotions using PyTorch.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from main import BrainwaveModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# TRAINING PARAMETERS - Configure these at the top
# =============================================================================
CHUNK_SIZE = 10                    # Size of each data chunk for training
EPOCHS = 50                        # Number of training epochs
BATCH_SIZE = 32                    # Training batch size
VALIDATION_SPLIT = 0.2             # Fraction of data for validation
TEST_SIZE = 0.2                    # Fraction of data for final testing
RANDOM_STATE = 42                  # Random seed for reproducibility

# Model hyperparameters
LEARNING_RATE = 0.001              # Learning rate for Adam optimizer
DROPOUT_RATE = 0.3                 # Dropout rate for regularization

# Data paths
DATA_DIR = '../backend/saved-audio'
MODEL_SAVE_PATH = '../models/brainwave_model.pth'

# Brainwave feature columns
FEATURE_COLUMNS = [
    'attention', 'meditation', 'delta', 'theta', 
    'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta'
]

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('../models', exist_ok=True)
    print("✓ Model directory created/verified")

def load_data_files():
    """Load all CSV and metadata files from the data directory."""
    print(f"Loading data files from {DATA_DIR}...")
    
    data_files = []
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        base_name = csv_file.replace('.csv', '')
        metadata_file = f'metadata_{base_name}.json'
        
        csv_path = os.path.join(DATA_DIR, csv_file)
        metadata_path = os.path.join(DATA_DIR, metadata_file)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            emotion_label = metadata['user_info']['target_emotion']
            data_files.append((csv_path, metadata_path, emotion_label))
    
    print(f"✓ Found {len(data_files)} data files with metadata")
    return data_files

def load_and_prepare_data():
    """Load all data files and prepare them for training."""
    data_files = load_data_files()
    if not data_files:
        raise ValueError("No data files found!")
    
    all_data = []
    all_labels = []
    
    for csv_path, metadata_path, emotion_label in data_files:
        print(f"Processing {os.path.basename(csv_path)} (emotion: {emotion_label})")
        
        df = pd.read_csv(csv_path)
        
        # Select only the feature columns
        df = df[FEATURE_COLUMNS]
        
        # Remove rows where all values are 0
        df = df[~(df == 0).all(axis=1)]
        
        # Handle NaN values
        df = df.fillna(0)
        
        if len(df) > 0:
            all_data.append(df)
            all_labels.append(emotion_label)
            print(f"  ✓ Loaded {len(df)} samples")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nCombined dataset summary:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Files processed: {len(all_data)}")
    print(f"  Emotion distribution: {pd.Series(all_labels).value_counts().to_dict()}")
    
    return combined_df, all_labels

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("BRAINWAVE CNN EMOTION CLASSIFICATION TRAINING")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    
    try:
        # Setup
        setup_directories()
        
        # Load data
        train_df, train_labels = load_and_prepare_data()
        
        # Initialize model
        model = BrainwaveModel(
            num_channels=len(FEATURE_COLUMNS), 
            num_points=CHUNK_SIZE,
            dropout_rate=DROPOUT_RATE
        )
        
        print(f"\n✓ CNN Model initialized:")
        print(f"  Channels: {len(FEATURE_COLUMNS)}")
        print(f"  Chunk size: {CHUNK_SIZE}")
        print(f"  Training data samples: {len(train_df)}")
        
        # Train the model
        print(f"\n🚀 Training CNN model for {EPOCHS} epochs...")
        model.train_model(
            df=train_df, 
            labels=train_labels, 
            epochs=EPOCHS, 
            validation_split=VALIDATION_SPLIT,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        
        # Save the model
        model.save_model(MODEL_SAVE_PATH)
        
        print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
        print(f"Classes trained: {list(model.label_encoder.classes_)}")
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    main()