#!/usr/bin/env python3
"""
Brain Wave Inference Script
Loads a trained model and performs emotion inference on brainwave data.
"""

import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
from settings import *
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = MODEL_SAVE_PATH
ENCODER_PATH = ENCODER_SAVE_PATH

# =============================================================================
# GLOBAL MODEL LOADING
# =============================================================================
print("Loading trained model and label encoder...")

# Load the trained model
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")

# Load the label encoder
if os.path.exists(ENCODER_PATH):
    label_encoder = joblib.load(ENCODER_PATH)
    print(f"✓ Label encoder loaded from {ENCODER_PATH}")
    print(f"✓ Available emotion classes: {list(label_encoder.classes_)}")
else:
    raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}. Please train the model first.")

print("✓ Inference system ready!\n")

# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def preprocess_dataframe(df):
    """
    Preprocess a dataframe for inference.
    
    Args:
        df (pd.DataFrame): Input dataframe with brainwave data
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Ensure we have all required columns
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select only the feature columns
    df = df[FEATURE_COLUMNS].copy()
    
    # Remove rows where all values are 0 (invalid readings)
    df = df[~(df == 0).all(axis=1)]
    
    # Handle any remaining NaN values
    df = df.fillna(0)
    
    return df


def create_inference_chunk(df, chunk_size=CHUNK_SIZE):
    """
    Create a single chunk from dataframe for inference.
    
    Args:
        df (pd.DataFrame): Input dataframe
        chunk_size (int): Size of chunk to create
        
    Returns:
        np.array: Single chunk with shape (1, chunk_size, num_features)
    """
    if len(df) < chunk_size:
        # Pad with zeros at the front if not enough data
        padding_needed = chunk_size - len(df)
        padding_df = pd.DataFrame(np.zeros((padding_needed, len(FEATURE_COLUMNS))), 
                                columns=FEATURE_COLUMNS)
        padded_df = pd.concat([padding_df, df], ignore_index=True)
        return np.array([padded_df.values])
    
    # Use the most recent data
    recent_data = df.tail(chunk_size)
    return np.array([recent_data.values])


def infer_emotion(df):
    """
    Infer emotion from brainwave data.
    
    Args:
        df (pd.DataFrame): DataFrame containing brainwave data with required columns
        
    Returns:
        str: Predicted emotion label
    """
    try:
        # Preprocess the dataframe
        processed_df = preprocess_dataframe(df)
        
        if len(processed_df) == 0:
            return "unknown"  # No valid data
        
        # If we have more data than chunk size, take only the latest chunk_size rows
        if len(processed_df) > CHUNK_SIZE:
            processed_df = processed_df.tail(CHUNK_SIZE)
        
        # Create inference chunk
        X = create_inference_chunk(processed_df, CHUNK_SIZE)
        
        # Make prediction
        prediction = model.predict(X, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        
        # Decode the prediction
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        print(f"Predicted emotion: {predicted_label} (confidence: {confidence:.3f})")
        return predicted_label
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return "error"


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_inference_on_file(csv_path, metadata_path=None):
    """
    Test inference on a single CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        metadata_path (str): Optional path to metadata file for ground truth
    """
    print(f"\n--- Testing: {os.path.basename(csv_path)} ---")
    
    # Load the CSV data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} data points")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Get ground truth if available
    ground_truth = None
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            ground_truth = metadata['user_info']['target_emotion']
            print(f"Ground truth: {ground_truth}")
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    # Make prediction
    predicted_emotion = infer_emotion(df)
    
    # Compare results
    if ground_truth:
        correct = predicted_emotion == ground_truth
        print(f"Result: {'✓ CORRECT' if correct else '✗ INCORRECT'}")
    else:
        print(f"Prediction: {predicted_emotion}")


def run_test_suite():
    """Run inference tests on available data files."""
    print("=" * 60)
    print("RUNNING INFERENCE TEST SUITE")
    print("=" * 60)
    
    data_dir = DATA_DIR
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found for testing!")
        return
    
    correct_predictions = 0
    total_predictions = 0
    
    for csv_file in csv_files[:5]:  # Test first 5 files
        # Find corresponding metadata file
        base_name = csv_file.replace('.csv', '')
        metadata_file = f'metadata_{base_name}.json'
        
        csv_path = os.path.join(data_dir, csv_file)
        metadata_path = os.path.join(data_dir, metadata_file)
        
        test_inference_on_file(csv_path, metadata_path if os.path.exists(metadata_path) else None)
        
        # Count accuracy if we have ground truth
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                ground_truth = metadata['user_info']['target_emotion']
                
                df = pd.read_csv(csv_path)
                predicted = infer_emotion(df)
                
                if predicted == ground_truth:
                    correct_predictions += 1
                total_predictions += 1
            except:
                pass
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n" + "=" * 60)
        print(f"TEST SUMMARY: {correct_predictions}/{total_predictions} correct ({accuracy:.2%})")
        print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_test_suite()
