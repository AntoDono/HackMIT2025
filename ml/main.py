import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class BrainwaveModel(nn.Module):
    def __init__(self, num_channels=8, num_points=100, num_classes=5, dropout_rate=0.3):
        """
        CNN-based brainwave model for EEG classification.
        
        Args:
            num_channels (int): Number of EEG channels (features)
            num_points (int): Number of time points per sample
            num_classes (int): Number of emotion classes
            dropout_rate (float): Dropout rate for regularization
        """
        super(BrainwaveModel, self).__init__()
        self.num_channels = num_channels
        self.num_points = num_points
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Temporal convolution for frequency filtering
        self.temporal_conv = nn.Conv2d(1, 16, kernel_size=(1, 25), padding=(0, 12), bias=False)
        self.temporal_bn = nn.BatchNorm2d(16)
        
        # Spatial convolution (depthwise)
        self.spatial_conv = nn.Conv2d(16, 32, kernel_size=(num_channels, 1), groups=16, bias=False)
        self.spatial_bn = nn.BatchNorm2d(32)
        
        # Separable convolution
        self.separable_conv = nn.Conv2d(32, 32, kernel_size=(1, 15), padding=(0, 7), bias=False)
        self.separable_bn = nn.BatchNorm2d(32)
        
        # Pooling and dropout
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate flattened size after convolutions
        self.flatten_size = self._get_flatten_size()
        
        # Fully connected layers
        self.fc = nn.Linear(self.flatten_size, num_classes)
        
    def _get_flatten_size(self):
        """Calculate the size after all conv and pooling layers"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.num_channels, self.num_points)
            x = self._forward_conv(x)
            return x.numel()
    
    def _forward_conv(self, x):
        """Forward pass through convolutional layers"""
        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = F.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout(x)
        
        # Separable convolution
        x = self.separable_conv(x)
        x = self.separable_bn(x)
        x = F.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout(x)
        
        return x
    
    def forward(self, x):
        """Forward pass through the entire network"""
        # Reshape input: (batch, channels, time) -> (batch, 1, channels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    
    def _create_chunks(self, df):
        """
        Create chunks of data from the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with columns for each wave type
            
        Returns:
            torch.Tensor: Tensor of chunks with shape (num_chunks, num_channels, num_points)
        """
        if len(df) < self.num_points:
            # Pad with zeros if not enough data
            padding_needed = self.num_points - len(df)
            padded_df = pd.concat([df, pd.DataFrame(np.zeros((padding_needed, self.num_channels)), 
                                                  columns=df.columns)])
            chunk = torch.FloatTensor(padded_df.values.T)  # Transpose: (channels, time)
            return chunk.unsqueeze(0)  # Add batch dimension
        
        chunks = []
        for i in range(len(df) - self.num_points + 1):
            chunk = df.iloc[i:i + self.num_points].values.T  # Transpose: (channels, time)
            chunks.append(torch.FloatTensor(chunk))
            
        return torch.stack(chunks)
    
    def train_model(self, df, labels, epochs=50, validation_split=0.2, batch_size=32, 
                   learning_rate=0.001, device=None):
        """
        Train the model on brain wave data.
        
        Args:
            df (pd.DataFrame): DataFrame containing wave data
            labels (list): List of emotion labels
            epochs (int): Number of training epochs
            validation_split (float): Fraction of data to use for validation
            batch_size (int): Training batch size
            learning_rate (float): Learning rate for optimizer
            device: PyTorch device (cuda/cpu)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(device)
        print(f"Training on device: {device}")
        print(f"Training model with {len(df)} samples...")
        
        # Create chunks from the data
        X = self._create_chunks(df)
        
        # Create labels for each chunk
        if isinstance(labels, str):
            labels = [labels] * len(X)
        elif len(labels) == 1:
            labels = labels * len(X)
        elif len(labels) != len(X):
            labels = list(labels) + [labels[-1]] * (len(X) - len(labels))
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels[:len(X)])
        y = torch.LongTensor(y)
        
        print(f"Model input shape: {X.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        # Split data
        n_val = int(len(X) * validation_split)
        indices = torch.randperm(len(X))
        
        X_train, X_val = X[indices[n_val:]], X[indices[:n_val]]
        y_train, y_val = y[indices[n_val:]], y[indices[:n_val]]
        
        # Create data loaders
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate averages
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.is_trained = True
        print("Training completed!")
        return None
    
    def predict(self, df, device=None):
        """
        Predict emotions from brain wave data.
        
        Args:
            df (pd.DataFrame): DataFrame containing wave data
            device: PyTorch device
            
        Returns:
            str: Predicted emotion label
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(device)
        self.eval()
        
        # Trim data if too long
        if len(df) > self.num_points:
            df = df.tail(self.num_points)
            print(f"Data trimmed to latest {self.num_points} points")
        
        # Create chunks
        X = self._create_chunks(df)
        X = X.to(device)
        
        with torch.no_grad():
            outputs = self(X[-1:])  # Use last chunk
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            confidence = torch.max(probabilities).item()
        
        # Decode prediction
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        print(f"Predicted emotion: {predicted_label} (confidence: {confidence:.3f})")
        return predicted_label
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'label_encoder': self.label_encoder,
            'num_channels': self.num_channels,
            'num_points': self.num_points,
            'num_classes': self.num_classes,
            'is_trained': self.is_trained
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath, device=None):
        """Load a pre-trained model."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.label_encoder = checkpoint['label_encoder']
        self.is_trained = checkpoint['is_trained']
        self.to(device)
        print(f"Model loaded from {filepath}")

def generate_fake_data(num_samples=2000, wave_types=['attention', 'meditation', 'delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta']):
    """Generate fake brain wave data for testing."""
    np.random.seed(42)
    
    data = {}
    for i, wave in enumerate(wave_types):
        # Generate different frequency patterns for different waves
        freq = 0.1 + i * 0.05  # Different frequencies for each wave
        data[wave] = np.random.normal(10, 2, num_samples) + 5 * np.sin(2 * np.pi * freq * np.arange(num_samples))
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("=== Brain Wave CNN Model Demo ===\n")
    
    # Initialize model
    wave_types = ['attention', 'meditation', 'delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta']
    num_points = 50
    model = BrainwaveModel(num_channels=len(wave_types), num_points=num_points, num_classes=4)
    
    print(f"Model initialized with {len(wave_types)} channels")
    print(f"Number of points per sample: {num_points}\n")
    
    # Generate fake training data
    print("Generating fake training data...")
    train_data = generate_fake_data(num_samples=500, wave_types=wave_types)
    train_labels = ['happy'] * 125 + ['sad'] * 125 + ['focused'] * 125 + ['relaxed'] * 125
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Number of labels: {len(train_labels)}")
    print(f"Label distribution: {pd.Series(train_labels).value_counts().to_dict()}\n")
    
    # Train the model
    print("Training the CNN model...")
    model.train_model(train_data, train_labels, epochs=10, validation_split=0.2)
    
    # Generate fake test data
    print("\nGenerating fake test data...")
    test_data = generate_fake_data(num_samples=100, wave_types=wave_types)
    print(f"Test data shape: {test_data.shape}\n")
    
    # Make predictions
    print("Making predictions...")
    prediction = model.predict(test_data.head(60))
    
    print(f"\nPrediction result: {prediction}")
    print("\n=== Demo completed successfully! ===")