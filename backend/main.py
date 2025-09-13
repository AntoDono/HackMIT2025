import os
import json
import threading
import socket
import argparse
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np

# External hooks (try/except imports)
try:
    import external_eeg_reader
    print("External EEG reader found")
except ImportError:
    external_eeg_reader = None
    print("Using internal EEG parsing")

try:
    import external_training
    print("External training module found")
except ImportError:
    external_training = None
    print("Using internal ML pipeline")

# Constants & schema
BRAINWAVE_COLUMNS = [
    "attention", "meditation", "delta", "theta", 
    "lowAlpha", "highAlpha", "lowBeta", "highBeta"
]
MODEL_PATH = "models/latest.joblib"

# Create directories if missing
Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
Path("saved-audio").mkdir(exist_ok=True)

# Simple validation function (replacing Pydantic)
def validate_brainwave_sample(data: Dict) -> Dict[str, float]:
    """Validate and convert brainwave sample data"""
    result = {}
    for col in BRAINWAVE_COLUMNS:
        if col not in data:
            raise ValueError(f"Missing required field: {col}")
        try:
            result[col] = float(data[col])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value for {col}: {data[col]}")
    return result

# In-memory store
df_lock = threading.RLock()
brainwave_data = pd.DataFrame(columns=BRAINWAVE_COLUMNS, dtype=float)

# Global model storage
cached_model = None
model_meta = {}

def add_row(sample_dict: Dict[str, float]) -> int:
    """Add a row to the DataFrame and return new size."""
    global brainwave_data
    with df_lock:
        # Ensure order matches BRAINWAVE_COLUMNS
        ordered_values = [sample_dict[col] for col in BRAINWAVE_COLUMNS]
        new_row = pd.DataFrame([ordered_values], columns=BRAINWAVE_COLUMNS)
        brainwave_data = pd.concat([brainwave_data, new_row], ignore_index=True)
        return len(brainwave_data)

def get_center_slice() -> pd.DataFrame:
    """Get center 100 rows (excluding first 10 and last 10)."""
    with df_lock:
        total_rows = len(brainwave_data)
        if total_rows < 110:  # Need at least 110 rows for 10 + 100 + 10
            raise ValueError(f"Insufficient data: {total_rows} rows, need at least 110 for center slice")
        return brainwave_data.iloc[10:110].copy()  # Rows 10-109 (100 rows total)

def size() -> int:
    """Get current size."""
    with df_lock:
        return len(brainwave_data)

def reset():
    """Reset the DataFrame."""
    global brainwave_data
    with df_lock:
        brainwave_data = pd.DataFrame(columns=BRAINWAVE_COLUMNS, dtype=float)

def save_csv(path: str):
    """Save DataFrame to CSV."""
    with df_lock:
        brainwave_data.to_csv(path, index=False)

def save_all_data():
    """Save all collected data to the saved-audio directory."""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("saved-audio")
    
    with df_lock:
        if len(brainwave_data) == 0:
            print("No data to save.")
            return
        
        # Save all collected data
        processed_data = brainwave_data.copy()
        
        # Save processed CSV data
        csv_path = save_dir / f"brainwave_data_{timestamp}.csv"
        processed_data.to_csv(csv_path, index=False)
        print(f"Saved {len(processed_data)} samples to {csv_path}")
        
        # Save model if it exists
        if cached_model is not None:
            model_save_path = save_dir / f"model_{timestamp}.joblib"
            import joblib
            model_bundle = {
                "pipeline": cached_model,
                "meta": model_meta
            }
            joblib.dump(model_bundle, model_save_path)
            print(f"Saved model to {model_save_path}")
        
        # Save metadata
        meta_path = save_dir / f"metadata_{timestamp}.json"
        metadata = {
            "timestamp": timestamp,
            "total_samples": len(processed_data),
            "columns": BRAINWAVE_COLUMNS,
            "model_meta": model_meta
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {meta_path}")

# External EEG reader adapter
def parse_eeg_frame(frame: Dict[str, Any]) -> Dict[str, float]:
    """Parse EEG frame using external reader if available."""
    if external_eeg_reader:
        # Try parse_eeg_frame function first
        if hasattr(external_eeg_reader, 'parse_eeg_frame'):
            return external_eeg_reader.parse_eeg_frame(frame)
        # Try ExternalEEGReader class
        elif hasattr(external_eeg_reader, 'ExternalEEGReader'):
            reader = external_eeg_reader.ExternalEEGReader()
            return reader.parse(frame)
    
    # Identity function - assume frame already matches schema
    return frame

def accept_eeg_data(sample_dict: Dict[str, Any]) -> Dict[str, float]:
    """Process EEG data through external reader if present."""
    parsed = parse_eeg_frame(sample_dict)
    
    # Ensure all required keys are present and are floats
    result = {}
    for col in BRAINWAVE_COLUMNS:
        if col not in parsed:
            raise ValueError(f"Missing required field: {col}")
        try:
            result[col] = float(parsed[col])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value for {col}: {parsed[col]}")
    
    return result

# ML helpers (with external fallbacks)
def internal_train_and_save(df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
    """Internal training pipeline."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline
    import joblib
    
    global cached_model, model_meta
    
    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=0, n_init=10))
    ])
    
    # Fit on the 8-column float data
    X = df[BRAINWAVE_COLUMNS].values
    pipeline.fit(X)
    
    # Save model
    model_bundle = {
        "pipeline": pipeline,
        "meta": {
            "n_samples_trained": len(df),
            "n_clusters": n_clusters,
            "columns": BRAINWAVE_COLUMNS.copy()
        }
    }
    
    joblib.dump(model_bundle, MODEL_PATH)
    
    # Update global state
    cached_model = pipeline
    model_meta = model_bundle["meta"]
    
    return {
        "n_samples_trained": len(df),
        "inertia": pipeline.named_steps['kmeans'].inertia_,
        "n_clusters": n_clusters,
        "columns": BRAINWAVE_COLUMNS.copy()
    }

def internal_load_model() -> bool:
    """Load model from disk."""
    global cached_model, model_meta
    
    if not os.path.exists(MODEL_PATH):
        return False
    
    try:
        import joblib
        model_bundle = joblib.load(MODEL_PATH)
        cached_model = model_bundle["pipeline"]
        model_meta = model_bundle["meta"]
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def train_and_save(df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
    """Train and save model (external preferred)."""
    if external_training and hasattr(external_training, 'train_and_save'):
        return external_training.train_and_save(df, n_clusters=n_clusters)
    else:
        return internal_train_and_save(df, n_clusters)

def load_model() -> bool:
    """Load model (external preferred)."""
    if external_training and hasattr(external_training, 'load_model'):
        return external_training.load_model()
    else:
        return internal_load_model()

def predict_one(sample_dict: Dict[str, float]) -> Dict[str, Any]:
    """Predict on a single sample."""
    global cached_model
    
    # Ensure model is loaded
    if cached_model is None:
        if not load_model():
            raise ValueError("No trained model available")
    
    # Prepare data
    X = np.array([[sample_dict[col] for col in BRAINWAVE_COLUMNS]])
    
    # Predict
    label = cached_model.predict(X)[0]
    distances = cached_model.transform(X)[0].tolist()
    
    return {
        "label": int(label),
        "distances": distances,
        "columns": BRAINWAVE_COLUMNS.copy()
    }

# Socket server functions
def handle_client_connection(client_socket, client_address):
    """Handle individual client connection"""
    print(f"Client connected from {client_address}")
    
    try:
        while True:
            # Receive data from client
            data = client_socket.recv(1024)
            if not data:
                break
                
            # Decode and process each line
            message = data.decode('utf-8')
            lines = message.strip().split('\n')
            
            for line in lines:
                if not line.strip():
                    continue
                    
                try:
                    # Parse JSON
                    frame = json.loads(line)
                    print(f"Received: {frame}")
                    
                    # Process through external reader if available
                    parsed_frame = accept_eeg_data(frame)
                    
                    # Validate data
                    sample_dict = validate_brainwave_sample(parsed_frame)
                    
                    # Add to DataFrame
                    new_size = add_row(sample_dict)
                    
                    # Send acknowledgment
                    response = json.dumps({"ok": True, "count": new_size}) + '\n'
                    client_socket.send(response.encode('utf-8'))
                    
                    print(f"Added sample, total rows: {new_size}")
                    
                except (json.JSONDecodeError, ValueError) as e:
                    error_response = json.dumps({"ok": False, "error": str(e)}) + '\n'
                    client_socket.send(error_response.encode('utf-8'))
                    print(f"Error processing data: {e}")
                    
    except Exception as e:
        print(f"Client {client_address} error: {e}")
    finally:
        client_socket.close()
        print(f"Client {client_address} disconnected")

def start_socket_server(host="0.0.0.0", port=8000, save_mode=False):
    """Start the socket server"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Flag to control server shutdown
    server_running = threading.Event()
    server_running.set()
    
    def input_handler():
        """Handle user input in save mode"""
        if save_mode:
            print("Type 'stop' to save data and exit...")
            while server_running.is_set():
                try:
                    user_input = input().strip().lower()
                    if user_input == 'stop':
                        print("Stopping server and saving data...")
                        save_all_data()
                        server_running.clear()
                        break
                except EOFError:
                    # Handle Ctrl+C or EOF
                    break
    
    # Start input handler thread if in save mode
    if save_mode:
        input_thread = threading.Thread(target=input_handler)
        input_thread.daemon = True
        input_thread.start()
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"EEG Socket Server listening on {host}:{port}")
        if save_mode:
            print("Save mode enabled. Type 'stop' to save data and exit.")
        
        # Set socket timeout to check server_running flag periodically
        server_socket.settimeout(1.0)
        
        while server_running.is_set():
            try:
                client_socket, client_address = server_socket.accept()
                
                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=handle_client_connection,
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except socket.timeout:
                # Check if we should continue running
                continue
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
        if save_mode:
            save_all_data()
    finally:
        server_running.clear()
        server_socket.close()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="EEG Socket Server")
    parser.add_argument("--save", action="store_true", 
                       help="Enable save mode - type 'stop' to save data and exit")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to bind to (default: 8000)")
    
    args = parser.parse_args()
    
    print("Starting EEG Socket Server...")
    if args.save:
        print("Save mode enabled.")
    
    start_socket_server(host=args.host, port=args.port, save_mode=args.save)

if __name__ == "__main__":
    main()