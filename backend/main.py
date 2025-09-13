import os
import json
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import uvicorn

# External hooks (try/except imports)
try:
    import external_eeg_reader
    print("✓ External EEG reader found")
except ImportError:
    external_eeg_reader = None
    print("ℹ Using internal EEG parsing")

try:
    import external_training
    print("✓ External training module found")
except ImportError:
    external_training = None
    print("ℹ Using internal ML pipeline")

# Constants & schema
BRAINWAVE_COLUMNS = [
    "attention", "meditation", "delta", "theta", 
    "lowAlpha", "highAlpha", "lowBeta", "highBeta"
]
MODEL_PATH = "models/latest.joblib"

# Create directories if missing
Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Pydantic models
class BrainwaveSample(BaseModel):
    attention: float
    meditation: float
    delta: float
    theta: float
    lowAlpha: float
    highAlpha: float
    lowBeta: float
    highBeta: float
    
    class Config:
        extra = 'forbid'

class TrainResponse(BaseModel):
    n_samples_trained: int
    inertia: float
    n_clusters: int
    columns: List[str]

class InferResponse(BaseModel):
    label: int
    distances: List[float]
    columns: List[str]

class ModelStatusResponse(BaseModel):
    loaded: bool
    n_samples_trained: Optional[int] = None
    n_clusters: Optional[int] = None
    columns: Optional[List[str]] = None
    path: str

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

def tail(n: int = 20) -> pd.DataFrame:
    """Get last n rows."""
    with df_lock:
        return brainwave_data.tail(n).copy()

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
            raise HTTPException(status_code=400, detail="No trained model available")
    
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

# FastAPI app
app = FastAPI(title="EEG Brainwave Backend", version="1.0.0")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"ok": True, "rows": size()}

@app.get("/data")
async def get_data(n: int = 20):
    """Get last n rows of data."""
    data_df = tail(n)
    return {
        "columns": BRAINWAVE_COLUMNS.copy(),
        "rows": data_df.values.tolist(),
        "count": len(data_df)
    }

@app.post("/train", response_model=TrainResponse)
async def train_model(min_samples: int = 100, n_clusters: int = 3):
    """Train the model."""
    current_size = size()
    
    if current_size < min_samples:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient data: {current_size} rows, need at least {min_samples}"
        )
    
    # Get current data
    df = tail(current_size)
    
    # Train model
    result = train_and_save(df, n_clusters)
    
    return TrainResponse(**result)

@app.post("/infer", response_model=InferResponse)
async def infer_emotion(sample: BrainwaveSample):
    """Infer emotion from brainwave data."""
    sample_dict = sample.dict()
    result = predict_one(sample_dict)
    return InferResponse(**result)

@app.get("/model/status", response_model=ModelStatusResponse)
async def model_status():
    """Get model status."""
    # Try to load model if not in memory
    if cached_model is None:
        load_model()
    
    return ModelStatusResponse(
        loaded=cached_model is not None,
        n_samples_trained=model_meta.get("n_samples_trained"),
        n_clusters=model_meta.get("n_clusters"),
        columns=model_meta.get("columns"),
        path=MODEL_PATH
    )

@app.websocket("/ws/brainwaves")
async def websocket_ingest(websocket: WebSocket):
    """WebSocket endpoint for EEG data ingestion only."""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            # Handle newline-delimited JSON
            lines = data.strip().split('\n')
            
            for line in lines:
                if not line.strip():
                    continue
                    
                try:
                    # Parse JSON
                    frame = json.loads(line)
                    
                    # Process through external reader if available
                    parsed_frame = accept_eeg_data(frame)
                    
                    # Validate with Pydantic
                    sample = BrainwaveSample(**parsed_frame)
                    
                    # Add to DataFrame
                    new_size = add_row(sample.dict())
                    
                    # Send acknowledgment
                    await websocket.send_text(json.dumps({
                        "ok": True,
                        "count": new_size
                    }))
                    
                except (json.JSONDecodeError, ValidationError, ValueError) as e:
                    await websocket.send_text(json.dumps({
                        "ok": False,
                        "error": str(e)
                    }))
                    
    except WebSocketDisconnect:
        print("WebSocket disconnected")

@app.websocket("/ws/brainwaves/train")
async def websocket_train(websocket: WebSocket):
    """WebSocket endpoint for EEG data ingestion with automatic training."""
    await websocket.accept()
    
    min_samples = 100  # Default threshold
    trained_this_session = False
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            # Handle newline-delimited JSON
            lines = data.strip().split('\n')
            
            for line in lines:
                if not line.strip():
                    continue
                    
                try:
                    # Parse JSON
                    frame = json.loads(line)
                    
                    # Process through external reader if available
                    parsed_frame = accept_eeg_data(frame)
                    
                    # Validate with Pydantic
                    sample = BrainwaveSample(**parsed_frame)
                    
                    # Add to DataFrame
                    new_size = add_row(sample.dict())
                    
                    # Send acknowledgment
                    await websocket.send_text(json.dumps({
                        "ok": True,
                        "count": new_size
                    }))
                    
                    # Check if we should train (only once per session)
                    if not trained_this_session and new_size >= min_samples:
                        trained_this_session = True
                        
                        # Train model
                        df = tail(new_size)
                        result = train_and_save(df, n_clusters=3)
                        
                        # Send training summary
                        await websocket.send_text(json.dumps({
                            "trained": True,
                            "n_samples": result["n_samples_trained"],
                            "inertia": result["inertia"],
                            "n_clusters": result["n_clusters"]
                        }))
                    
                except (json.JSONDecodeError, ValidationError, ValueError) as e:
                    await websocket.send_text(json.dumps({
                        "ok": False,
                        "error": str(e)
                    }))
                    
    except WebSocketDisconnect:
        print("WebSocket train disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)