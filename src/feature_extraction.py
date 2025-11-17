#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: feature_extraction.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Script for extracting statistical and temporal features from RFID tag data segments.
    Computes comprehensive features including RSSI statistics, circular phase statistics,
    and temporal sampling characteristics for machine learning and analysis purposes.

License: MIT License
Dependencies: numpy, sklearn, pandas, scipy
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt
import os

from timediff import split_tag_data_by_absolute_and_stat

def extract_features(tag_values):
    """
    Extracts statistical features from tag data.
    
    :param tag_values: Dictionary with 'timestamp', 'rssi', 'phase' (numpy arrays, phase in degrees from -180 to 180)
    :return: Dictionary with extracted features
    """
    timestamps = tag_values['timestamp']
    rssi = tag_values['rssi']
    phase = tag_values['phase']  # in degrees from -180 to 180

    # RSSI features
    rssi_mean = np.mean(rssi)
    rssi_std = np.std(rssi)
    rssi_min = np.min(rssi)
    rssi_max = np.max(rssi)

    # Phase features (circular mean and standard deviation in degrees)
    phase_mean = circmean(phase, high=180, low=-180)
    phase_std = circstd(phase, high=180, low=-180)
    num_samples = len(phase)

    # Temporal features
    time_diffs = np.diff(timestamps)
    total_time = timestamps[-1] - timestamps[0]
    if len(time_diffs) > 0:
        time_diff_mean = np.mean(time_diffs)
        time_diff_std = np.std(time_diffs)
        time_diff_min = np.min(time_diffs)
        time_diff_max = np.max(time_diffs)
    else:
        time_diff_mean = 0
        time_diff_std = 0
        time_diff_min = 0
        time_diff_max = 0

    features = {
        'rssi_mean': rssi_mean,
        'rssi_std': rssi_std,
        'rssi_min': rssi_min,
        'rssi_max': rssi_max,
        'phase_mean': phase_mean,
        'phase_std': phase_std,
        'num_samples': num_samples,
        'total_time': total_time,
        'time_diff_mean': time_diff_mean,
        'time_diff_std': time_diff_std,
        'time_diff_min': time_diff_min,
        'time_diff_max': time_diff_max
    }

    return features

def normalize_features(features_list):
    """
    Normalizes a list of feature dictionaries using MinMaxScaler.
    
    :param features_list: List of dictionaries (output from extract_features)
    :return: Normalized numpy array of features and the fitted scaler
    """
    df = pd.DataFrame(features_list)
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(df)
    return features_normalized, scaler

def plot_feature_comparison(features_list, labels=None, output_dir='output_plots'):
    """
    Creates comparison plots of extracted features.
    
    :param features_list: List of feature dictionaries
    :param labels: List of labels for each feature set (optional)
    :param output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(features_list)
    
    if labels is None:
        labels = [f'Segment {i+1}' for i in range(len(features_list))]
    
    # Create feature comparison plots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    feature_names = list(df.columns)
    colors = plt.cm.tab10(np.linspace(0, 1, len(features_list)))
    
    for i, feature in enumerate(feature_names):
        ax = axes[i]
        
        # Bar plot for each feature
        x_pos = np.arange(len(features_list))
        values = df[feature].values
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Segments')
        ax.set_ylabel(feature.replace('_', ' ').title())
        ax.set_title(f'{feature.replace("_", " ").title()} Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'S{i+1}' for i in range(len(features_list))], rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_correlation_matrix(features_list, output_dir='output_plots'):
    """
    Creates a correlation matrix heatmap of features.
    
    :param features_list: List of feature dictionaries
    :param output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(features_list)
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(12, 10))
    im = plt.imshow(correlation_matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    plt.colorbar(im, shrink=0.8)
    
    # Add labels
    feature_names = [name.replace('_', '\n') for name in correlation_matrix.columns]
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.yticks(range(len(feature_names)), feature_names)
    
    # Add correlation values as text
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_statistics(features_list):
    """
    Analyzes and prints statistical information about extracted features.
    
    :param features_list: List of feature dictionaries
    """
    df = pd.DataFrame(features_list)
    
    print("=== FEATURE STATISTICS ANALYSIS ===\n")
    
    print(f"Number of samples: {len(features_list)}")
    print(f"Number of features: {len(df.columns)}")
    
    print("\n=== DESCRIPTIVE STATISTICS ===")
    stats = df.describe()
    print(stats.round(4))
    
    print("\n=== FEATURE RANGES ===")
    for feature in df.columns:
        min_val = df[feature].min()
        max_val = df[feature].max()
        range_val = max_val - min_val
        print(f"{feature:15}: [{min_val:8.3f}, {max_val:8.3f}] (range: {range_val:8.3f})")
    
    print("\n=== FEATURE VARIABILITY (Coefficient of Variation) ===")
    for feature in df.columns:
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        cv = (std_val / abs(mean_val)) * 100 if mean_val != 0 else 0
        print(f"{feature:15}: {cv:6.2f}% (std/mean)")

def main():
    """
    Main function demonstrating the use of feature extraction functions.
    """

    from csv_data_loader import extract_tag_data

    print("=== RFID TAG FEATURE EXTRACTION DEMONSTRATION ===\n")
    
    try:
  
        # Path to CSV file
        csv_path = '../data/dynamic.csv'
        
        if os.path.exists(csv_path):
            print("Loading real RFID data...")
            
            # Load data
            tag_data_dict = extract_tag_data(csv_path)
            
            if not tag_data_dict:
                raise Exception("No data found in CSV file")
            
            # Get first tag
            tag_id = list(tag_data_dict.keys())[0]
            tag_data = tag_data_dict[tag_id]
            
            print(f"Loaded tag: {tag_id}")
            print(f"Total samples: {len(tag_data['timestamp'])}")
            
            # Segment the data
            print("\nSegmenting data...")
            segments = split_tag_data_by_absolute_and_stat(tag_data, abs_threshold=1.0, stat_threshold=2.0)
            
            print(f"Found {len(segments)} segments")
            
            # Extract features from first 5 segments (or all if less than 5)
            max_segments = min(5, len(segments))
            features_list = []
            
            print(f"\nExtracting features from {max_segments} segments...")
            
            for i in range(max_segments):
                segment = segments[i]
                features = extract_features(segment)
                features_list.append(features)
                
                print(f"Segment {i+1}: {len(segment['timestamp'])} samples, "
                      f"duration: {segment['timestamp'][-1] - segment['timestamp'][0]:.2f}s")
            
            real_data = True
            
        else:
            raise Exception("CSV file not found")
            
    except Exception as e:
        print(f"Could not load real data ({e}), using synthetic data for demonstration...")
        real_data = False
        
        # Generate synthetic RFID-like data
        print("\nGenerating synthetic RFID data...")
        
        features_list = []
        np.random.seed(42)  # For reproducible results
        
        for i in range(5):
            # Generate synthetic segment
            n_samples = np.random.randint(50, 200)
            duration = np.random.uniform(0.5, 3.0)
            
            # Timestamps
            timestamps = np.linspace(0, duration, n_samples)
            
            # RSSI (typically negative values, with some noise)
            base_rssi = np.random.uniform(-70, -30)
            rssi_noise = np.random.normal(0, 2, n_samples)
            rssi = np.full(n_samples, base_rssi) + rssi_noise
            
            # Phase (circular data between -180 and 180)
            base_phase = np.random.uniform(-180, 180)
            phase_trend = np.linspace(0, np.random.uniform(-50, 50), n_samples)
            phase_noise = np.random.normal(0, 10, n_samples)
            phase = np.mod(base_phase + phase_trend + phase_noise + 180, 360) - 180
            
            # Create segment dictionary
            segment_data = {
                'timestamp': timestamps,
                'rssi': rssi,
                'phase': phase
            }
            
            # Extract features
            features = extract_features(segment_data)
            features_list.append(features)
            
            print(f"Synthetic segment {i+1}: {n_samples} samples, duration: {duration:.2f}s")
    
    # Analyze extracted features
    print("\n" + "="*60)
    print("ANALYZING EXTRACTED FEATURES")
    print("="*60)
    
    analyze_feature_statistics(features_list)
    
    # Normalize features
    print("\n" + "="*60)
    print("NORMALIZING FEATURES")
    print("="*60)
    
    features_normalized, scaler = normalize_features(features_list)
    
    print(f"Original feature matrix shape: {len(features_list)} x {len(features_list[0])}")
    print(f"Normalized feature matrix shape: {features_normalized.shape}")
    
    print("\nNormalization scaler info:")
    feature_names = list(features_list[0].keys())
    for i, feature_name in enumerate(feature_names):
        print(f"{feature_name:15}: scale={scaler.scale_[i]:.6f}, min={scaler.min_[i]:.6f}")
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Plot feature comparisons
    segment_labels = [f'Segment {i+1}' for i in range(len(features_list))]
    plot_feature_comparison(features_list, segment_labels)
    
    # Plot correlation matrix
    if len(features_list) > 1:
        plot_feature_correlation_matrix(features_list)
    else:
        print("Need at least 2 segments to compute correlation matrix")
    
    # Display normalized features
    print("\n=== NORMALIZED FEATURES ===")
    print("First 3 samples of normalized features:")
    normalized_df = pd.DataFrame(features_normalized[:3], columns=feature_names)
    print(normalized_df.round(4))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    data_source = "real RFID data" if real_data else "synthetic data"
    print(f"Successfully processed {len(features_list)} segments from {data_source}")
    print(f"Extracted {len(feature_names)} features per segment:")
    
    for i, feature_name in enumerate(feature_names):
        print(f"  {i+1:2d}. {feature_name}")
    
    print(f"\nOutput files saved to 'output_plots/' directory:")
    print("  - feature_comparison.png: Bar plots comparing features across segments")
    if len(features_list) > 1:
        print("  - feature_correlation.png: Correlation matrix heatmap")
    
    print(f"\nFeatures are ready for machine learning applications!")
    
    return features_list, features_normalized, scaler

if __name__ == "__main__":
    features_list, features_normalized, scaler = main()

