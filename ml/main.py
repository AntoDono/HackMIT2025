import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class BrainwaveModel:
    def __init__(self, wave_types=['alpha', 'beta', 'gamma', 'delta', 'theta'], num_points=100):
        """
        Initialize the brain wave model.
        
        Args:
            wave_types (list): List of wave type names (e.g., ['alpha', 'beta', 'gamma'])
            num_points (int): Number of data points to use for each prediction/training sample
        """
        self.wave_types = wave_types
        self.num_points = num_points
        self.num_waves = len(wave_types)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def _build_model(self, num_classes, hidden_units=None, dropout_rate=0.3, learning_rate=0.001):
        """Build a neural network model with configurable parameters."""
        if hidden_units is None:
            hidden_units = [128, 64, 32]
        
        model = keras.Sequential([
            keras.layers.Input(shape=(self.num_points, self.num_waves)),
            keras.layers.Flatten(),
        ])
        
        # Add configurable hidden layers
        for units in hidden_units:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        
        # Compile with configurable optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_chunks(self, df):
        """
        Create chunks of data from the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with columns for each wave type
            
        Returns:
            np.array: Array of chunks with shape (num_chunks, num_points, num_waves)
        """
        if len(df) < self.num_points:
            # If we don't have enough data, pad with zeros or repeat the data
            padding_needed = self.num_points - len(df)
            padded_df = pd.concat([df, pd.DataFrame(np.zeros((padding_needed, len(self.wave_types))), 
                                                  columns=self.wave_types)])
            return np.array([padded_df.values])
        
        num_chunks = len(df) - self.num_points + 1
        chunks = []
        
        for i in range(num_chunks):
            chunk = df.iloc[i:i + self.num_points][self.wave_types].values
            chunks.append(chunk)
            
        return np.array(chunks)
    
    def train(self, df, labels, epochs=50, validation_split=0.2, batch_size=32, 
              hidden_units=None, dropout_rate=0.3, learning_rate=0.001, 
              early_stopping_patience=10, reduce_lr_patience=5):
        """
        Train the model on brain wave data with configurable hyperparameters.
        
        Args:
            df (pd.DataFrame): DataFrame containing wave data with columns matching wave_types
            labels (list): List of emotion labels (e.g., ['happy', 'sad', 'depressed'])
            epochs (int): Number of training epochs
            validation_split (float): Fraction of data to use for validation
            batch_size (int): Training batch size
            hidden_units (list): List of hidden layer sizes (e.g., [128, 64, 32])
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for Adam optimizer
            early_stopping_patience (int): Patience for early stopping callback
            reduce_lr_patience (int): Patience for learning rate reduction callback
        """
        print(f"Training model with {len(df)} samples...")
        print(f"Hyperparameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Hidden units: {hidden_units or [128, 64, 32]}")
        print(f"  Validation split: {validation_split}")
        
        # Create chunks from the data
        X = self._create_chunks(df)
        
        # Create labels for each chunk (assuming the same label applies to all chunks from one sample)
        if isinstance(labels, str):
            labels = [labels] * len(X)
        elif len(labels) == 1:
            labels = labels * len(X)
        elif len(labels) != len(X):
            # If we have fewer labels than chunks, repeat the last label
            labels = list(labels) + [labels[-1]] * (len(X) - len(labels))
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels[:len(X)])
        
        # Build model with custom hyperparameters
        num_classes = len(self.label_encoder.classes_)
        self.model = self._build_model(
            num_classes=num_classes,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        print(f"Model input shape: {X.shape}")
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Model parameters: {self.model.count_params():,}")
        
        # Setup callbacks
        callbacks = []
        
        if early_stopping_patience > 0:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ))
        
        if reduce_lr_patience > 0:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            ))
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1,
            batch_size=batch_size,
            callbacks=callbacks if callbacks else None
        )
        
        self.is_trained = True
        print("Training completed!")
        return history
    
    def predict(self, df):
        """
        Predict emotions from brain wave data.
        
        Args:
            df (pd.DataFrame): DataFrame containing wave data
            
        Returns:
            str: Predicted emotion label
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        # Trim data if it's too long (keep the latest points)
        if len(df) > self.num_points:
            df = df.tail(self.num_points)
            print(f"Data trimmed to latest {self.num_points} points")
        
        # Create chunks
        X = self._create_chunks(df)
        
        # Make prediction on the last chunk (most recent data)
        prediction = self.model.predict(X[-1:], verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Decode the prediction
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        confidence = np.max(prediction)
        
        print(f"Predicted emotion: {predicted_label} (confidence: {confidence:.3f})")
        return predicted_label
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model."""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

def generate_fake_data(num_samples=2000, wave_types=['alpha', 'beta', 'gamma', 'delta', 'theta']):
    """Generate fake brain wave data for testing."""
    np.random.seed(42)  # For reproducible results
    
    data = {}
    for wave in wave_types:
        # Generate different patterns for different waves
        if wave == 'alpha':
            data[wave] = np.random.normal(10, 2, num_samples) + np.sin(np.linspace(0, 20, num_samples))
        elif wave == 'beta':
            data[wave] = np.random.normal(15, 3, num_samples) + np.cos(np.linspace(0, 15, num_samples))
        elif wave == 'gamma':
            data[wave] = np.random.normal(5, 1, num_samples) + 0.5 * np.sin(np.linspace(0, 30, num_samples))
        elif wave == 'delta':
            data[wave] = np.random.normal(8, 1.5, num_samples) + np.sin(np.linspace(0, 5, num_samples))
        elif wave == 'theta':
            data[wave] = np.random.normal(12, 2.5, num_samples) + np.cos(np.linspace(0, 10, num_samples))
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("=== Brain Wave Model Demo ===\n")
    
    # Initialize model
    wave_types = ['alpha', 'beta', 'gamma', 'delta', 'theta']
    num_points = 50  # Use 50 points per prediction for faster demo
    model = BrainwaveModel(wave_types=wave_types, num_points=num_points)
    
    print(f"Model initialized with waves: {wave_types}")
    print(f"Number of points per sample: {num_points}\n")
    
    # Generate fake training data
    print("Generating fake training data...")
    train_data = generate_fake_data(num_samples=500, wave_types=wave_types)  # Smaller for demo
    train_labels = ['happy'] * 150 + ['sad'] * 150 + ['depressed'] * 100 + ['excited'] * 100
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Number of labels: {len(train_labels)}")
    print(f"Label distribution: {pd.Series(train_labels).value_counts().to_dict()}\n")
    
    # Train the model
    print("Training the model...")
    history = model.train(train_data, train_labels, epochs=10, validation_split=0.2)
    
    # Generate fake test data
    print("\nGenerating fake test data...")
    test_data = generate_fake_data(num_samples=100, wave_types=wave_types)
    print(f"Test data shape: {test_data.shape}\n")
    
    # Make predictions
    print("Making predictions...")
    prediction1 = model.predict(test_data.head(60))  # Normal size
    prediction2 = model.predict(test_data.tail(30))  # Smaller than num_points
    prediction3 = model.predict(test_data)  # Larger than num_points (will be trimmed)
    
    print(f"\nPrediction results:")
    print(f"1. First 60 samples: {prediction1}")
    print(f"2. Last 30 samples: {prediction2}")
    print(f"3. All 100 samples (trimmed): {prediction3}")
    
    print("\n=== Demo completed successfully! ===")