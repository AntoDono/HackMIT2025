import pandas as pd
import numpy as np
import time
from main import BrainwaveModel, generate_fake_data

def test_basic_functionality():
    """Test basic model functionality (original demo)."""
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
    return model

def test_realtime_inference(model, chunk_sizes=[10, 50, 100, 200, 500]):
    """Test real-time inference with different chunk sizes and timing."""
    print("\n=== Real-Time Inference Performance Test ===\n")
    
    wave_types = model.wave_types
    results = []
    
    for chunk_size in chunk_sizes:
        print(f"Testing chunk size: {chunk_size}")
        
        # Generate test data for this chunk size
        test_data = generate_fake_data(num_samples=chunk_size, wave_types=wave_types)
        
        # Measure inference time
        start_time = time.time()
        prediction = model.predict(test_data)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        results.append({
            'chunk_size': chunk_size,
            'inference_time_ms': inference_time,
            'prediction': prediction
        })
        
        print(f"  Chunk size: {chunk_size} | Time: {inference_time:.2f}ms | Prediction: {prediction}")
    
    print(f"\n=== Performance Summary ===")
    print("Chunk Size | Inference Time (ms)")
    print("-" * 35)
    for result in results:
        print(f"{result['chunk_size']:10d} | {result['inference_time_ms']:15.2f}")
    
    # Calculate average time per sample
    print(f"\nAverage inference times:")
    for result in results:
        avg_per_sample = result['inference_time_ms'] / result['chunk_size']
        print(f"Chunk {result['chunk_size']:3d}: {avg_per_sample:.4f}ms per sample")
    
    return results

if __name__ == "__main__":
    # Run basic functionality test and get trained model
    model = test_basic_functionality()
    
    # Run real-time inference performance test
    performance_results = test_realtime_inference(model)
    
    print("\n=== All tests completed! ===")
