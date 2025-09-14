#!/usr/bin/env python3
"""
EEG Statistical Analysis Module
Provides comprehensive statistical analysis of brainwave data for pattern recognition and insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Brainwave frequency bands (Hz)
FREQUENCY_BANDS = {
    'delta': (0.5, 4),      # Deep sleep, healing
    'theta': (4, 8),        # REM sleep, creativity, meditation
    'alpha': (8, 13),       # Relaxation, calm alertness
    'beta': (13, 30),       # Active thinking, focus
    'gamma': (30, 100)      # High-level cognitive processing
}

# Expected brainwave columns
BRAINWAVE_COLUMNS = [
    "attention", "meditation", "delta", "theta", 
    "lowAlpha", "highAlpha", "lowBeta", "highBeta", "lowGamma", "midGamma"
]

def calculate_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic statistical measures for all brainwave channels."""
    stats = {}
    
    for column in BRAINWAVE_COLUMNS:
        if column in df.columns:
            data = df[column].dropna()
            if len(data) > 0:
                stats[column] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'range': float(data.max() - data.min()),
                    'variance': float(data.var()),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'percentile_25': float(data.quantile(0.25)),
                    'percentile_75': float(data.quantile(0.75)),
                    'iqr': float(data.quantile(0.75) - data.quantile(0.25))
                }
    
    return stats

def calculate_wave_ratios(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate important brainwave ratios for cognitive assessment."""
    ratios = {}
    
    try:
        # Alpha/Theta ratio (cognitive performance indicator)
        if 'lowAlpha' in df.columns and 'theta' in df.columns:
            alpha_mean = (df['lowAlpha'] + df.get('highAlpha', 0)).mean()
            theta_mean = df['theta'].mean()
            if theta_mean > 0:
                ratios['alpha_theta_ratio'] = float(alpha_mean / theta_mean)
        
        # Beta/Alpha ratio (arousal/relaxation balance)
        if 'lowBeta' in df.columns and 'lowAlpha' in df.columns:
            beta_mean = (df['lowBeta'] + df.get('highBeta', 0)).mean()
            alpha_mean = (df['lowAlpha'] + df.get('highAlpha', 0)).mean()
            if alpha_mean > 0:
                ratios['beta_alpha_ratio'] = float(beta_mean / alpha_mean)
        
        # Theta/Beta ratio (ADHD indicator)
        if 'theta' in df.columns and 'lowBeta' in df.columns:
            theta_mean = df['theta'].mean()
            beta_mean = (df['lowBeta'] + df.get('highBeta', 0)).mean()
            if beta_mean > 0:
                ratios['theta_beta_ratio'] = float(theta_mean / beta_mean)
        
        # Attention/Meditation balance
        if 'attention' in df.columns and 'meditation' in df.columns:
            att_mean = df['attention'].mean()
            med_mean = df['meditation'].mean()
            if med_mean > 0:
                ratios['attention_meditation_ratio'] = float(att_mean / med_mean)
        
        # High/Low frequency ratio
        high_freq = df[['lowBeta', 'highBeta']].mean().mean() if all(col in df.columns for col in ['lowBeta', 'highBeta']) else 0
        low_freq = df[['delta', 'theta']].mean().mean() if all(col in df.columns for col in ['delta', 'theta']) else 0
        if low_freq > 0:
            ratios['high_low_freq_ratio'] = float(high_freq / low_freq)
            
    except Exception as e:
        print(f"Error calculating ratios: {e}")
    
    return ratios

def calculate_coherence_measures(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate coherence and synchronization measures between brainwave bands."""
    coherence = {}
    
    try:
        # Calculate correlation coefficients between different bands
        wave_pairs = [
            ('lowAlpha', 'highAlpha'),
            ('lowBeta', 'highBeta'),
            ('delta', 'theta'),
            ('attention', 'meditation'),
            ('lowAlpha', 'meditation'),
            ('lowBeta', 'attention')
        ]
        
        for wave1, wave2 in wave_pairs:
            if wave1 in df.columns and wave2 in df.columns:
                corr = df[wave1].corr(df[wave2])
                if not np.isnan(corr):
                    coherence[f'{wave1}_{wave2}_correlation'] = float(corr)
        
        # Calculate overall brainwave coherence (average correlation)
        correlations = []
        for col1 in BRAINWAVE_COLUMNS:
            for col2 in BRAINWAVE_COLUMNS:
                if col1 != col2 and col1 in df.columns and col2 in df.columns:
                    corr = df[col1].corr(df[col2])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if correlations:
            coherence['overall_coherence'] = float(np.mean(correlations))
            coherence['coherence_std'] = float(np.std(correlations))
    
    except Exception as e:
        print(f"Error calculating coherence: {e}")
    
    return coherence

def calculate_temporal_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate temporal features and trends in the data."""
    temporal = {}
    
    try:
        # Data quality metrics
        temporal['total_samples'] = len(df)
        temporal['missing_data_percentage'] = float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
        
        # Calculate trends for each brainwave
        for column in BRAINWAVE_COLUMNS:
            if column in df.columns and len(df) > 1:
                # Simple linear trend
                x = np.arange(len(df))
                y = df[column].values
                
                # Remove NaN values
                valid_indices = ~np.isnan(y)
                if np.sum(valid_indices) > 1:
                    x_valid = x[valid_indices]
                    y_valid = y[valid_indices]
                    
                    # Calculate slope (trend)
                    slope = np.polyfit(x_valid, y_valid, 1)[0]
                    temporal[f'{column}_trend'] = float(slope)
                    
                    # Calculate variability (coefficient of variation)
                    if y_valid.mean() != 0:
                        temporal[f'{column}_cv'] = float(y_valid.std() / abs(y_valid.mean()))
        
        # Session stability (how consistent the readings are)
        if len(df) > 10:
            # Calculate rolling standard deviation
            window_size = min(10, len(df) // 2)
            stabilities = []
            
            for column in BRAINWAVE_COLUMNS:
                if column in df.columns:
                    rolling_std = df[column].rolling(window=window_size).std().dropna()
                    if len(rolling_std) > 0:
                        stabilities.append(rolling_std.mean())
            
            if stabilities:
                temporal['session_stability'] = float(np.mean(stabilities))
    
    except Exception as e:
        print(f"Error calculating temporal features: {e}")
    
    return temporal

def calculate_cognitive_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate cognitive performance indicators based on brainwave patterns."""
    cognitive = {}
    
    try:
        # Focus index (combination of attention and beta waves)
        if 'attention' in df.columns and 'lowBeta' in df.columns:
            attention_norm = df['attention'] / 100.0  # Normalize to 0-1
            beta_norm = df['lowBeta'] / df['lowBeta'].max() if df['lowBeta'].max() > 0 else 0
            focus_scores = (attention_norm + beta_norm) / 2
            cognitive['focus_index'] = float(focus_scores.mean())
            cognitive['focus_consistency'] = float(1 - focus_scores.std())
        
        # Relaxation index (combination of meditation and alpha waves)
        if 'meditation' in df.columns and 'lowAlpha' in df.columns:
            meditation_norm = df['meditation'] / 100.0
            alpha_norm = df['lowAlpha'] / df['lowAlpha'].max() if df['lowAlpha'].max() > 0 else 0
            relax_scores = (meditation_norm + alpha_norm) / 2
            cognitive['relaxation_index'] = float(relax_scores.mean())
            cognitive['relaxation_consistency'] = float(1 - relax_scores.std())
        
        # Mental workload indicator (high beta activity)
        if 'highBeta' in df.columns:
            workload_norm = df['highBeta'] / df['highBeta'].max() if df['highBeta'].max() > 0 else 0
            cognitive['mental_workload'] = float(workload_norm.mean())
        
        # Drowsiness indicator (high theta + delta activity)
        if 'theta' in df.columns and 'delta' in df.columns:
            theta_norm = df['theta'] / df['theta'].max() if df['theta'].max() > 0 else 0
            delta_norm = df['delta'] / df['delta'].max() if df['delta'].max() > 0 else 0
            drowsiness_scores = (theta_norm + delta_norm) / 2
            cognitive['drowsiness_index'] = float(drowsiness_scores.mean())
        
        # Cognitive arousal (beta/alpha ratio over time)
        if 'lowBeta' in df.columns and 'lowAlpha' in df.columns:
            beta_vals = df['lowBeta'] + df.get('highBeta', 0)
            alpha_vals = df['lowAlpha'] + df.get('highAlpha', 0)
            arousal_ratios = beta_vals / (alpha_vals + 1e-6)  # Add small epsilon to avoid division by zero
            cognitive['cognitive_arousal'] = float(arousal_ratios.mean())
            cognitive['arousal_variability'] = float(arousal_ratios.std())
    
    except Exception as e:
        print(f"Error calculating cognitive indicators: {e}")
    
    return cognitive

def analyze_eeg_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive EEG analysis function that returns all statistical measures.
    
    Args:
        df (pd.DataFrame): DataFrame containing EEG brainwave data
        
    Returns:
        Dict[str, Any]: Comprehensive statistics dictionary
    """
    if df is None or len(df) == 0:
        return {'error': 'Empty or invalid DataFrame provided'}
    
    # Validate required columns
    available_columns = [col for col in BRAINWAVE_COLUMNS if col in df.columns]
    if len(available_columns) == 0:
        return {'error': 'No valid brainwave columns found in DataFrame'}
    
    analysis = {
        'metadata': {
            'total_samples': len(df),
            'duration_minutes': len(df) / 60 if len(df) > 60 else len(df) / 60,  # Assuming 1 sample per second
            'available_channels': available_columns,
            'missing_channels': [col for col in BRAINWAVE_COLUMNS if col not in df.columns]
        }
    }
    
    # Calculate all statistical measures
    analysis['basic_stats'] = calculate_basic_stats(df)
    analysis['wave_ratios'] = calculate_wave_ratios(df)
    analysis['coherence'] = calculate_coherence_measures(df)
    analysis['temporal_features'] = calculate_temporal_features(df)
    analysis['cognitive_indicators'] = calculate_cognitive_indicators(df)
    
    # Add summary insights
    analysis['summary'] = generate_summary_insights(analysis)
    
    return analysis

def generate_summary_insights(analysis: Dict[str, Any]) -> Dict[str, str]:
    """Generate human-readable insights from the statistical analysis."""
    insights = {}
    
    try:
        # Focus assessment
        focus_index = analysis.get('cognitive_indicators', {}).get('focus_index', 0)
        if focus_index > 0.7:
            insights['focus_level'] = 'High focus and concentration detected'
        elif focus_index > 0.4:
            insights['focus_level'] = 'Moderate focus levels'
        else:
            insights['focus_level'] = 'Low focus, may indicate distraction or fatigue'
        
        # Relaxation assessment
        relax_index = analysis.get('cognitive_indicators', {}).get('relaxation_index', 0)
        if relax_index > 0.7:
            insights['relaxation_level'] = 'High relaxation and calm state'
        elif relax_index > 0.4:
            insights['relaxation_level'] = 'Moderate relaxation levels'
        else:
            insights['relaxation_level'] = 'Low relaxation, possible stress or tension'
        
        # Mental workload
        workload = analysis.get('cognitive_indicators', {}).get('mental_workload', 0)
        if workload > 0.7:
            insights['mental_workload'] = 'High mental workload detected'
        elif workload > 0.4:
            insights['mental_workload'] = 'Moderate cognitive load'
        else:
            insights['mental_workload'] = 'Low mental workload'
        
        # Session quality
        stability = analysis.get('temporal_features', {}).get('session_stability', 0)
        if stability < 1000:  # Lower values indicate more stability
            insights['session_quality'] = 'Stable and consistent brainwave patterns'
        else:
            insights['session_quality'] = 'Variable brainwave patterns, possible artifacts'
        
        # Overall coherence
        coherence = analysis.get('coherence', {}).get('overall_coherence', 0)
        if coherence > 0.6:
            insights['brain_coherence'] = 'High brainwave coherence - good neural synchronization'
        elif coherence > 0.3:
            insights['brain_coherence'] = 'Moderate brainwave coherence'
        else:
            insights['brain_coherence'] = 'Low brainwave coherence - possible fragmented attention'
    
    except Exception as e:
        insights['error'] = f'Error generating insights: {e}'
    
    return insights

# Convenience function for quick analysis
def quick_eeg_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick analysis returning only the most important metrics."""
    full_analysis = analyze_eeg_patterns(df)
    
    quick_stats = {
        'sample_count': full_analysis.get('metadata', {}).get('total_samples', 0),
        'focus_index': full_analysis.get('cognitive_indicators', {}).get('focus_index', 0),
        'relaxation_index': full_analysis.get('cognitive_indicators', {}).get('relaxation_index', 0),
        'attention_avg': full_analysis.get('basic_stats', {}).get('attention', {}).get('mean', 0),
        'meditation_avg': full_analysis.get('basic_stats', {}).get('meditation', {}).get('mean', 0),
        'alpha_theta_ratio': full_analysis.get('wave_ratios', {}).get('alpha_theta_ratio', 0),
        'overall_coherence': full_analysis.get('coherence', {}).get('overall_coherence', 0),
        'primary_insight': list(full_analysis.get('summary', {}).values())[0] if full_analysis.get('summary') else 'No insights available'
    }
    
    return quick_stats

if __name__ == "__main__":
    # Example usage and testing
    print("EEG Statistics Analysis Module")
    print("=" * 40)
    
    # Create sample data for testing
    import random
    
    sample_data = {
        'attention': [random.randint(30, 90) for _ in range(100)],
        'meditation': [random.randint(20, 80) for _ in range(100)],
        'delta': [random.randint(50000, 200000) for _ in range(100)],
        'theta': [random.randint(40000, 180000) for _ in range(100)],
        'lowAlpha': [random.randint(20000, 80000) for _ in range(100)],
        'highAlpha': [random.randint(15000, 70000) for _ in range(100)],
        'lowBeta': [random.randint(10000, 50000) for _ in range(100)],
        'highBeta': [random.randint(5000, 30000) for _ in range(100)]
    }
    
    test_df = pd.DataFrame(sample_data)
    
    # Run analysis
    results = analyze_eeg_patterns(test_df)
    
    print("Sample Analysis Results:")
    print(f"Total samples: {results['metadata']['total_samples']}")
    print(f"Focus index: {results['cognitive_indicators'].get('focus_index', 'N/A'):.3f}")
    print(f"Relaxation index: {results['cognitive_indicators'].get('relaxation_index', 'N/A'):.3f}")
    print(f"Alpha/Theta ratio: {results['wave_ratios'].get('alpha_theta_ratio', 'N/A'):.3f}")
    print(f"Overall coherence: {results['coherence'].get('overall_coherence', 'N/A'):.3f}")
    
    print("\nInsights:")
    for key, value in results['summary'].items():
        print(f"- {key}: {value}")
