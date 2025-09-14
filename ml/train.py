#!/usr/bin/env python3
"""
Brain Wave Training Script
Trains a neural network model on brainwave data to classify emotions using the existing BrainwaveModel class.
"""

import os
import json
import pandas as pd
import numpy as np
from main import BrainwaveModel  # Use the existing model class
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# TRAINING PARAMETERS - Configure these at the top
# =============================================================================
CHUNK_SIZE = 50                    # Size of each data chunk for training (num_points in BrainwaveModel)
EPOCHS = 100                        # Number of training epochs
BATCH_SIZE = 32                    # Training batch size
VALIDATION_SPLIT = 0.2             # Fraction of data for validation
TEST_SIZE = 0.2                    # Fraction of data for final testing
RANDOM_STATE = 42                  # Random seed for reproducibility

# Model hyperparameters
LEARNING_RATE = 0.001              # Learning rate for Adam optimizer
DROPOUT_RATE = 0.05                 # Dropout rate for regularization
HIDDEN_UNITS = [256, 512, 1024, 512, 256]       # Hidden layer sizes
EARLY_STOPPING_PATIENCE = 10       # Patience for early stopping
REDUCE_LR_PATIENCE = 5             # Patience for learning rate reduction

# Data paths
DATA_DIR = '../backend/saved-audio'
MODEL_SAVE_PATH = '../models/brainwave_model.h5'

# Brainwave feature columns (matching the actual data format)
FEATURE_COLUMNS = [
    'attention', 'meditation', 'delta', 'theta', 
    'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta'
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('../models', exist_ok=True)
    print("✓ Model directory created/verified")


def load_data_files():
    """
    Load all CSV and metadata files from the data directory.
    
    Returns:
        list: List of tuples (csv_path, metadata_path, emotion_label)
    """
    print(f"Loading data files from {DATA_DIR}...")
    
    data_files = []
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        # Find corresponding metadata file
        base_name = csv_file.replace('.csv', '')
        metadata_file = f'metadata_{base_name}.json'
        
        csv_path = os.path.join(DATA_DIR, csv_file)
        metadata_path = os.path.join(DATA_DIR, metadata_file)
        
        if os.path.exists(metadata_path):
            # Load metadata to get emotion label
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            emotion_label = metadata['user_info']['target_emotion']
            data_files.append((csv_path, metadata_path, emotion_label))
        else:
            print(f"⚠️  No metadata found for {csv_file}")
    
    print(f"✓ Found {len(data_files)} data files with metadata")
    return data_files


def load_and_prepare_data():
    """
    Load all data files and prepare them for training with proper train/test split.
    
    Returns:
        tuple: (train_df, test_df, train_labels, test_labels, all_emotions)
    """
    print("Loading and preparing all data...")
    
    data_files = load_data_files()
    if not data_files:
        raise ValueError("No data files found!")
    
    all_data = []
    all_labels = []
    
    # Process each data file
    for csv_path, metadata_path, emotion_label in data_files:
        print(f"Processing {os.path.basename(csv_path)} (emotion: {emotion_label})")
        
        df = pd.read_csv(csv_path)
        
        # Ensure we have all required columns
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing columns in {csv_path}: {missing_cols}")
            continue
        
        # Select only the feature columns
        df = df[FEATURE_COLUMNS]
        
        # Remove rows where all values are 0 (invalid readings)
        df = df[~(df == 0).all(axis=1)]
        
        # Handle any remaining NaN values
        df = df.fillna(0)
        
        if len(df) > 0:
            all_data.append(df)
            all_labels.append(emotion_label)
            print(f"  ✓ Loaded {len(df)} samples")
        else:
            print(f"  ⚠️  No valid data in {csv_path}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    all_emotions = pd.Series(all_labels).value_counts().to_dict()
    
    print(f"\nCombined dataset summary:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Features: {list(combined_df.columns)}")
    print(f"  Files processed: {len(all_data)}")
    print(f"  Emotion distribution: {all_emotions}")
    
    # Split data into train and test sets
    # We'll split by files to ensure proper separation
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    
    # Group data by emotion for stratified split
    emotion_data = {}
    for i, emotion in enumerate(all_labels):
        if emotion not in emotion_data:
            emotion_data[emotion] = []
        emotion_data[emotion].append((all_data[i], emotion))
    
    # Split each emotion group
    for emotion, data_list in emotion_data.items():
        n_test = max(1, int(len(data_list) * TEST_SIZE))  # At least 1 for test
        
        # Randomly select test files
        np.random.seed(RANDOM_STATE)
        test_indices = np.random.choice(len(data_list), n_test, replace=False)
        
        for i, (data, label) in enumerate(data_list):
            if i in test_indices:
                test_data.append(data)
                test_labels.append(label)
            else:
                train_data.append(data)
                train_labels.append(label)
    
    # Combine train and test data
    train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
    test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
    
    print(f"\nData split summary:")
    print(f"  Training samples: {len(train_df)} ({len(train_labels)} files)")
    print(f"  Test samples: {len(test_df)} ({len(test_labels)} files)")
    print(f"  Train emotion distribution: {pd.Series(train_labels).value_counts().to_dict()}")
    print(f"  Test emotion distribution: {pd.Series(test_labels).value_counts().to_dict()}")
    
    return train_df, test_df, train_labels, test_labels, all_emotions


def evaluate_model_performance(model, test_df, test_labels):
    """
    Evaluate the trained model on test data with detailed metrics.
    
    Args:
        model: Trained BrainwaveModel instance
        test_df: Test dataframe
        test_labels: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 50)
    print("MODEL EVALUATION ON TEST DATA")
    print("=" * 50)
    
    if len(test_df) == 0:
        print("⚠️  No test data available for evaluation")
        return {}
    
    # Make predictions on test data
    predictions = []
    actual_labels = []
    
    # Create chunks from test data similar to training
    # We'll use the model's internal chunking method
    print(f"Evaluating on {len(test_labels)} test files...")
    
    for i, label in enumerate(test_labels):
        # Get a sample of test data for this label
        # Since we split by files, we need to predict on each file separately
        print(f"Testing file {i+1}/{len(test_labels)} (emotion: {label})")
        
        # For evaluation, we'll use the test data directly
        # The BrainwaveModel.predict() method handles chunking internally
        try:
            # Get a reasonable sample size for prediction
            sample_size = min(len(test_df), CHUNK_SIZE * 3)  # Get enough for chunking
            test_sample = test_df.sample(n=sample_size, random_state=RANDOM_STATE + i)
            
            predicted_emotion = model.predict(test_sample)
            predictions.append(predicted_emotion)
            actual_labels.append(label)
            
        except Exception as e:
            print(f"⚠️  Error predicting for {label}: {e}")
            continue
    
    if not predictions:
        print("❌ No successful predictions made")
        return {}
    
    # Calculate metrics
    accuracy = accuracy_score(actual_labels, predictions)
    
    print(f"\n✓ Evaluation completed!")
    print(f"  Test samples: {len(predictions)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
    print("-" * 50)
    class_report = classification_report(actual_labels, predictions, output_dict=True)
    print(classification_report(actual_labels, predictions))
    
    # Confusion Matrix
    print(f"\n📈 CONFUSION MATRIX:")
    print("-" * 30)
    cm = confusion_matrix(actual_labels, predictions)
    
    # Get unique labels for matrix display
    unique_labels = sorted(list(set(actual_labels + predictions)))
    
    print(f"Labels: {unique_labels}")
    print("Confusion Matrix (rows=actual, cols=predicted):")
    
    # Create a formatted confusion matrix
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    print(cm_df)
    
    # Per-class statistics
    print(f"\n📋 PER-CLASS STATISTICS:")
    print("-" * 40)
    for emotion in unique_labels:
        if emotion in class_report:
            metrics = class_report[emotion]
            print(f"{emotion:>12}: precision={metrics['precision']:.3f}, "
                  f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}, "
                  f"support={int(metrics['support'])}")
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print("-" * 30)
    macro_avg = class_report['macro avg']
    weighted_avg = class_report['weighted avg']
    
    print(f"{'Macro Average':>15}: precision={macro_avg['precision']:.3f}, "
          f"recall={macro_avg['recall']:.3f}, f1={macro_avg['f1-score']:.3f}")
    print(f"{'Weighted Average':>15}: precision={weighted_avg['precision']:.3f}, "
          f"recall={weighted_avg['recall']:.3f}, f1={weighted_avg['f1-score']:.3f}")
    
    # Prediction distribution
    print(f"\n📈 PREDICTION DISTRIBUTION:")
    print("-" * 35)
    pred_dist = pd.Series(predictions).value_counts()
    actual_dist = pd.Series(actual_labels).value_counts()
    
    comparison_df = pd.DataFrame({
        'Actual': actual_dist,
        'Predicted': pred_dist
    }).fillna(0).astype(int)
    
    print(comparison_df)
    
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'predictions': predictions,
        'actual_labels': actual_labels,
        'unique_labels': unique_labels
    }


def main():
    """Main training pipeline using the existing BrainwaveModel class."""
    print("=" * 60)
    print("BRAINWAVE EMOTION CLASSIFICATION TRAINING")
    print("Using existing BrainwaveModel class from main.py")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    
    try:
        # Setup
        setup_directories()
        
        # Load and prepare data with proper train/test split
        train_df, test_df, train_labels, test_labels, emotion_distribution = load_and_prepare_data()
        
        # Initialize the BrainwaveModel with the correct wave types and chunk size
        wave_types = FEATURE_COLUMNS
        model = BrainwaveModel(wave_types=wave_types, num_points=CHUNK_SIZE)
        
        print(f"\n✓ BrainwaveModel initialized:")
        print(f"  Wave types: {wave_types}")
        print(f"  Chunk size (num_points): {CHUNK_SIZE}")
        print(f"  Training data samples: {len(train_df)}")
        print(f"  Test data samples: {len(test_df)}")
        
        # Train the model using the existing train method with all hyperparameters
        print(f"\n🚀 Training model for {EPOCHS} epochs...")
        history = model.train(
            df=train_df, 
            labels=train_labels, 
            epochs=EPOCHS, 
            validation_split=VALIDATION_SPLIT,
            batch_size=BATCH_SIZE,
            hidden_units=HIDDEN_UNITS,
            dropout_rate=DROPOUT_RATE,
            learning_rate=LEARNING_RATE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            reduce_lr_patience=REDUCE_LR_PATIENCE
        )
        
        # Save the trained model
        model.save_model(MODEL_SAVE_PATH)
        
        print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
        print(f"Classes trained: {list(model.label_encoder.classes_)}")
        
        # Comprehensive model evaluation
        evaluation_results = evaluate_model_performance(model, test_df, test_labels)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 TRAINING AND EVALUATION COMPLETED!")
        print("=" * 60)
        
        if evaluation_results:
            print(f"📈 Final Test Accuracy: {evaluation_results['accuracy']:.4f} ({evaluation_results['accuracy']*100:.2f}%)")
            print(f"🎯 Emotions Classified: {evaluation_results['unique_labels']}")
            print(f"📊 Total Test Predictions: {len(evaluation_results['predictions'])}")
        
        print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
        print(f"🧠 Architecture: {len(FEATURE_COLUMNS)} features → chunks of {CHUNK_SIZE} → emotions")
        print(f"📁 Data distribution: {emotion_distribution}")
        
        # Quick verification test
        print(f"\n🔍 Quick verification test...")
        if len(test_df) > CHUNK_SIZE:
            test_sample = test_df.head(CHUNK_SIZE * 2)
            prediction = model.predict(test_sample)
            print(f"✓ Sample prediction successful: {prediction}")
        else:
            print("⚠️  Not enough test data for verification")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
