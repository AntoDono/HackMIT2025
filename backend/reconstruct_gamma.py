#!/usr/bin/env python3
"""
Gamma Wave Reconstruction Script
Reconstructs missing lowGamma and midGamma columns for legacy EEG data
based on neuroscience-backed relationships with other brainwave frequencies.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

def analyze_gamma_relationships(reference_file: str) -> Dict[str, float]:
    """
    Analyze gamma relationships from a reference file with complete data.
    Returns coefficients for gamma reconstruction.
    """
    df = pd.read_csv(reference_file)
    
    # Filter out zero/invalid samples
    valid_mask = (df['highBeta'] > 0) & (df['lowGamma'] > 0) & (df['midGamma'] > 0)
    valid_df = df[valid_mask]
    
    if len(valid_df) < 5:
        print(f"Warning: Only {len(valid_df)} valid samples in reference file")
        return get_default_coefficients()
    
    # Calculate relationships
    relationships = {}
    
    # Low Gamma relationships (30-40 Hz)
    # Typically correlated with high beta and attention
    relationships['lowGamma_to_highBeta'] = np.median(valid_df['lowGamma'] / valid_df['highBeta'])
    relationships['lowGamma_to_attention'] = np.median(valid_df['lowGamma'] / (valid_df['attention'] + 1))
    relationships['lowGamma_to_lowBeta'] = np.median(valid_df['lowGamma'] / (valid_df['lowBeta'] + 1))
    
    # Mid Gamma relationships (40-60 Hz) 
    # Usually lower than low gamma, correlated with cognitive processing
    relationships['midGamma_to_lowGamma'] = np.median(valid_df['midGamma'] / valid_df['lowGamma'])
    relationships['midGamma_to_highBeta'] = np.median(valid_df['midGamma'] / valid_df['highBeta'])
    relationships['midGamma_to_attention'] = np.median(valid_df['midGamma'] / (valid_df['attention'] + 1))
    
    # Cross-frequency coupling (gamma often follows beta patterns)
    relationships['gamma_beta_coupling'] = np.median((valid_df['lowGamma'] + valid_df['midGamma']) / 
                                                   (valid_df['lowBeta'] + valid_df['highBeta'] + 1))
    
    print("Analyzed gamma relationships:")
    for key, value in relationships.items():
        print(f"  {key}: {value:.4f}")
    
    return relationships

def get_default_coefficients() -> Dict[str, float]:
    """Default coefficients based on neuroscience literature."""
    return {
        'lowGamma_to_highBeta': 0.6,      # Low gamma ~60% of high beta
        'lowGamma_to_attention': 0.3,      # Moderate attention correlation  
        'lowGamma_to_lowBeta': 0.4,        # Lower correlation with low beta
        'midGamma_to_lowGamma': 0.7,       # Mid gamma ~70% of low gamma
        'midGamma_to_highBeta': 0.4,       # Lower correlation with high beta
        'midGamma_to_attention': 0.2,      # Weaker attention correlation
        'gamma_beta_coupling': 0.5         # Overall coupling strength
    }

def reconstruct_gamma_values(df: pd.DataFrame, coefficients: Dict[str, float]) -> pd.DataFrame:
    """
    Reconstruct gamma values based on existing brainwave data.
    """
    df_reconstructed = df.copy()
    
    # Add noise factor for more realistic variation
    np.random.seed(42)  # For reproducible results
    
    for idx in range(len(df_reconstructed)):
        row = df_reconstructed.iloc[idx]
        
        # Reconstruct lowGamma using multiple approaches
        lowGamma_estimates = []
        
        # Method 1: High Beta correlation
        if row['highBeta'] > 0:
            lowGamma_estimates.append(row['highBeta'] * coefficients['lowGamma_to_highBeta'])
        
        # Method 2: Attention correlation (when attention > 0)
        if row['attention'] > 0:
            lowGamma_estimates.append(row['attention'] * coefficients['lowGamma_to_attention'] * 100)
        
        # Method 3: Low Beta correlation
        if row['lowBeta'] > 0:
            lowGamma_estimates.append(row['lowBeta'] * coefficients['lowGamma_to_lowBeta'])
        
        # Method 4: Combined beta approach
        beta_total = row['lowBeta'] + row['highBeta']
        if beta_total > 0:
            lowGamma_estimates.append(beta_total * coefficients['gamma_beta_coupling'] * 0.6)
        
        # Take median of estimates for stability
        if lowGamma_estimates:
            base_lowGamma = np.median(lowGamma_estimates)
        else:
            base_lowGamma = 100  # Minimal baseline
        
        # Add realistic noise (±20%)
        noise_factor = 1 + np.random.normal(0, 0.15)
        reconstructed_lowGamma = max(0, base_lowGamma * noise_factor)
        
        # Reconstruct midGamma
        base_midGamma = reconstructed_lowGamma * coefficients['midGamma_to_lowGamma']
        
        # Mid gamma typically has more variation
        noise_factor = 1 + np.random.normal(0, 0.2)
        reconstructed_midGamma = max(0, base_midGamma * noise_factor)
        
        # Ensure gamma values are reasonable relative to other frequencies
        # Gamma should generally be lower than beta frequencies
        max_reasonable_gamma = (row['lowBeta'] + row['highBeta']) * 0.8
        if max_reasonable_gamma > 0:
            reconstructed_lowGamma = min(reconstructed_lowGamma, max_reasonable_gamma)
            reconstructed_midGamma = min(reconstructed_midGamma, max_reasonable_gamma * 0.7)
        
        df_reconstructed.at[idx, 'lowGamma'] = reconstructed_lowGamma
        df_reconstructed.at[idx, 'midGamma'] = reconstructed_midGamma
    
    return df_reconstructed

def process_file(input_file: str, output_file: str, coefficients: Dict[str, float]) -> bool:
    """Process a single CSV file to add gamma columns."""
    try:
        print(f"Processing {input_file}...")
        
        # Read the 8-column data
        df = pd.read_csv(input_file)
        
        # Verify it's the old format
        expected_cols = ["attention", "meditation", "delta", "theta", 
                        "lowAlpha", "highAlpha", "lowBeta", "highBeta"]
        
        if list(df.columns) != expected_cols:
            print(f"Unexpected columns in {input_file}: {list(df.columns)}")
            return False
        
        # Reconstruct gamma values
        df_with_gamma = reconstruct_gamma_values(df, coefficients)
        
        # Add the new columns
        df_with_gamma['lowGamma'] = df_with_gamma['lowGamma']
        df_with_gamma['midGamma'] = df_with_gamma['midGamma']
        
        # Reorder columns to match expected format
        final_columns = ["attention", "meditation", "delta", "theta", 
                        "lowAlpha", "highAlpha", "lowBeta", "highBeta", 
                        "lowGamma", "midGamma"]
        df_final = df_with_gamma[final_columns]
        
        # Save the reconstructed data
        df_final.to_csv(output_file, index=False)
        
        print(f"  ✓ Saved {len(df_final)} samples with reconstructed gamma to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def update_metadata_file(metadata_file: str) -> bool:
    """Update metadata JSON to include gamma columns."""
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Update columns list
        if 'columns' in metadata:
            expected_old = ["attention", "meditation", "delta", "theta", 
                           "lowAlpha", "highAlpha", "lowBeta", "highBeta"]
            if metadata['columns'] == expected_old:
                metadata['columns'] = expected_old + ["lowGamma", "midGamma"]
                metadata['gamma_reconstructed'] = True
                metadata['reconstruction_method'] = "beta_correlation_with_noise"
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"  ✓ Updated metadata: {metadata_file}")
                return True
        
        return False
        
    except Exception as e:
        print(f"Error updating metadata {metadata_file}: {e}")
        return False

def main():
    """Main reconstruction process."""
    print("🧠 EEG Gamma Wave Reconstruction Tool")
    print("=" * 50)
    
    # Directories
    no_gamma_dir = Path("saved-audio-no-gamma")
    reference_file = Path("saved-audio/jenniferhsu_focused_20250914_001804.csv")
    
    if not no_gamma_dir.exists():
        print(f"Directory {no_gamma_dir} not found!")
        return
    
    if not reference_file.exists():
        print(f"Reference file {reference_file} not found!")
        print("Using default coefficients based on neuroscience literature.")
        coefficients = get_default_coefficients()
    else:
        print(f"Analyzing gamma relationships from {reference_file}...")
        coefficients = analyze_gamma_relationships(str(reference_file))
    
    # Find all CSV files in no-gamma directory
    csv_files = list(no_gamma_dir.glob("*.csv"))
    print(f"\nFound {len(csv_files)} CSV files to process")
    
    success_count = 0
    
    for csv_file in csv_files:
        # Create output filename (move to main saved-audio directory)
        output_file = Path("saved-audio") / csv_file.name
        
        # Process the file
        if process_file(str(csv_file), str(output_file), coefficients):
            success_count += 1
            
            # Update corresponding metadata file
            metadata_file = no_gamma_dir / f"metadata_{csv_file.stem}.json"
            output_metadata = Path("saved-audio") / f"metadata_{csv_file.stem}.json"
            
            if metadata_file.exists():
                # Copy and update metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Update columns and add reconstruction info
                metadata['columns'] = ["attention", "meditation", "delta", "theta", 
                                     "lowAlpha", "highAlpha", "lowBeta", "highBeta", 
                                     "lowGamma", "midGamma"]
                metadata['gamma_reconstructed'] = True
                metadata['reconstruction_method'] = "beta_correlation_with_noise"
                metadata['reconstruction_coefficients'] = coefficients
                
                with open(output_metadata, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"  ✓ Updated metadata: {output_metadata}")
    
    print(f"\n✅ Successfully reconstructed {success_count}/{len(csv_files)} files")
    print("\nReconstructed files are now in the 'saved-audio' directory with gamma columns.")
    print("Original files remain in 'saved-audio-no-gamma' as backup.")

if __name__ == "__main__":
    main()
