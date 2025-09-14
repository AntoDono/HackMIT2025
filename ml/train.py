#!/usr/bin/env python3
"""
Brain Wave Training Script
Trains a neural network model on brainwave data to classify emotions.
"""

import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
from main import BrainwaveModel
warnings.filterwarnings('ignore')

# =============================================================================
# TRAINING PARAMETERS - Configure these at the top
# =============================================================================
CHUNK_SIZE = 30                    # Size of each data chunk for training
EPOCHS = 100                        # Number of training epochs
BATCH_SIZE = 64                    # Training batch size
VALIDATION_SPLIT = 0.2             # Fraction of data for validation
LEARNING_RATE = 0.001              # Learning rate for optimizer
DROPOUT_RATE = 0.05                 # Dropout rate for regularization
HIDDEN_UNITS = [512, 1024, 2048, 1024, 512]       # Hidden layer sizes
TEST_SIZE = 0.2                    # Fraction of data for testing
RANDOM_STATE = 42                  # Random seed for reproducibility

# Data paths
DATA_DIR = '../backend/saved-audio'
MODEL_SAVE_PATH = '../models/brainwave_model.h5'
ENCODER_SAVE_PATH = '../models/label_encoder.joblib'

# Brainwave feature columns
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


def load_single_file(csv_path):
    """
    Load and preprocess a single CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = pd.read_csv(csv_path)
    
    # Ensure we have all required columns
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Missing columns in {csv_path}: {missing_cols}")
        return None
    
    # Select only the feature columns
    df = df[FEATURE_COLUMNS]
    
    # Remove rows where all values are 0 (invalid readings)
    df = df[~(df == 0).all(axis=1)]
    
    # Handle any remaining NaN values
    df = df.fillna(0)
    
    return df


def apply_masking(data, mask_ratio=0.2):
    """
    Apply random masking to input data by setting mask_ratio of values to 0.
    
    Args:
        data (np.array): Input data array
        mask_ratio (float): Fraction of data to mask (set to 0)
        
    Returns:
        np.array: Masked data array
    """
    masked_data = data.copy()
    
    # For each sample in the batch
    for i in range(len(masked_data)):
        # Get the shape of this sample
        sample_shape = masked_data[i].shape
        total_elements = np.prod(sample_shape)
        
        # Calculate number of elements to mask
        num_to_mask = int(total_elements * mask_ratio)
        
        # Flatten the sample to easily select random indices
        flat_sample = masked_data[i].flatten()
        
        # Randomly select indices to mask
        mask_indices = np.random.choice(total_elements, num_to_mask, replace=False)
        
        # Set selected indices to 0
        flat_sample[mask_indices] = 0
        
        # Reshape back to original shape
        masked_data[i] = flat_sample.reshape(sample_shape)
    
    return masked_data


def create_chunks(df, chunk_size=CHUNK_SIZE):
    """
    Create overlapping chunks from the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        chunk_size (int): Size of each chunk
        
    Returns:
        np.array: Array of chunks with shape (num_chunks, chunk_size, num_features)
    """
    if len(df) < chunk_size:
        # Pad with zeros if not enough data
        padding_needed = chunk_size - len(df)
        padded_df = pd.concat([
            df, 
            pd.DataFrame(np.zeros((padding_needed, len(FEATURE_COLUMNS))), 
                        columns=FEATURE_COLUMNS)
        ])
        return np.array([padded_df.values])
    
    # Create overlapping chunks
    chunks = []
    for i in range(len(df) - chunk_size + 1):
        chunk = df.iloc[i:i + chunk_size].values
        chunks.append(chunk)
    
    return np.array(chunks)


def prepare_training_data():
    """
    Load all data files and prepare training datasets.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoder)
    """
    print("Preparing training data...")
    
    data_files = load_data_files()
    if not data_files:
        raise ValueError("No data files found!")
    
    all_chunks = []
    all_labels = []
    
    # Process each data file
    for csv_path, metadata_path, emotion_label in data_files:
        print(f"Processing {os.path.basename(csv_path)} (emotion: {emotion_label})")
        
        df = load_single_file(csv_path)
        if df is None:
            continue
        
        # Create chunks from this file
        chunks = create_chunks(df, CHUNK_SIZE)
        
        # Add chunks and labels
        all_chunks.extend(chunks)
        all_labels.extend([emotion_label] * len(chunks))
        
        print(f"  ✓ Created {len(chunks)} chunks")
    
    # Convert to numpy arrays
    X = np.array(all_chunks)
    y = np.array(all_labels)
    
    print(f"\nDataset summary:")
    print(f"  Total chunks: {len(X)}")
    print(f"  Chunk shape: {X[0].shape}")
    print(f"  Emotion distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Hardcode the labels in a specific order
    hardcoded_labels = ['focused', 'nervous', 'relax', 'stress', 'surprise']
    
    # Encode labels with hardcoded classes
    label_encoder = LabelEncoder()
    label_encoder.fit(hardcoded_labels)  # Fit with hardcoded labels first
    y_encoded = label_encoder.transform(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        stratify=y_encoded
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, label_encoder

def train_model(model, X_train, y_train):
    """
    Train the model.
    
    Args:
        model: Keras model to train
        X_train: Training features
        y_train: Training labels
        
    Returns:
        keras.callbacks.History: Training history
    """
    print(f"Training model for {EPOCHS} epochs...")
    
    # Apply masking to training data (20% of each input set to 0)
    print("Applying masking to training data (20% of values set to 0)...")
    X_train_masked = apply_masking(X_train, mask_ratio=0.2)
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
    ]
    
    # Train the model with masked data
    history = model.fit(
        X_train_masked, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✓ Training completed!")
    return history


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder for decoding predictions
    """
    print("Evaluating model on test data...")
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_test)
    print(f"✓ Test accuracy: {accuracy:.4f}")
    
    # Classification report
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    
    return accuracy


def save_model_and_encoder(model, label_encoder):
    """
    Save the trained model and label encoder.
    
    Args:
        model: Trained Keras model
        label_encoder: Fitted label encoder
    """
    print("Saving model and encoder...")
    
    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"✓ Model saved to {MODEL_SAVE_PATH}")
    
    # Save label encoder
    joblib.dump(label_encoder, ENCODER_SAVE_PATH)
    print(f"✓ Label encoder saved to {ENCODER_SAVE_PATH}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("BRAINWAVE EMOTION CLASSIFICATION TRAINING")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    try:
        # Setup
        setup_directories()
        
        # Prepare data
        X_train, X_test, y_train, y_test, label_encoder = prepare_training_data()
        
        # Build model
        brainwave_model = BrainwaveModel(
            wave_types=FEATURE_COLUMNS,
            num_points=CHUNK_SIZE,
            hidden_units=HIDDEN_UNITS,
            dropout_rate=DROPOUT_RATE,
            learning_rate=LEARNING_RATE
        )
        num_classes = len(label_encoder.classes_)
        model = brainwave_model._build_model(num_classes)
        
        print(f"✓ Model built with {model.count_params()} parameters")
        model.summary()
        
        # Train model
        history = train_model(model, X_train, y_train)
        
        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test, label_encoder)
        
        # Save model
        save_model_and_encoder(model, label_encoder)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final test accuracy: {accuracy:.4f}")
        print(f"Model saved to: {MODEL_SAVE_PATH}")
        print(f"Label encoder saved to: {ENCODER_SAVE_PATH}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
