#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: curve_mean.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Script for computing averaged patterns from RFID tag segments and comparing 
    individual segments against the mean pattern. Provides statistical analysis
    of segment similarity using correlation, MSE, and z-score metrics.

License: MIT License
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap

from timediff import split_tag_data_by_absolute_and_stat

def compute_averaged_pattern(segments, feature='rssi', num_points=100, normalize_method='minmax'):
    """
    Computes the normalized average pattern and its variance from multiple segments.
    
    :param segments: List of dictionaries with 'timestamp', 'rssi', 'phase'
    :param feature: Feature to process ('rssi' or 'phase')
    :param num_points: Number of points for common interpolation
    :param normalize_method: 'max' (by maximum) or 'minmax' (0-1)
    :return: dict with 'time_normalized', 'mean_pattern', 'variance_pattern', 'std_pattern'
    """
    
    if not segments:
        return None
    
    normalized_patterns = []
    
    for segment in segments:
        if len(segment['timestamp']) < 2:
            continue
            
        # Extract time and measurements
        time = segment['timestamp']
        values = segment[feature]
        
        # Normalize time (0 to 1)
        time_range = time.max() - time.min()
        if time_range == 0:
            # If all timestamps are equal (1 sample), skip
            print(f"Warning: Segment with constant time, skipping...")
            continue
        time_norm = (time - time.min()) / time_range
        
        # Normalize values according to method
        if normalize_method == 'max':
            # For RSSI (negative values): normalize by the least negative value (absolute maximum)
            if feature == 'rssi':
                max_val = np.max(values)
                if max_val == 0:
                    # If maximum is 0, use minimum absolute value
                    values_norm = values / (np.min(np.abs(values)) + 1e-8)
                else:
                    values_norm = values / max_val
            else:
                max_abs = np.max(np.abs(values))
                if max_abs == 0:
                    values_norm = np.zeros_like(values)  # All values are 0
                else:
                    values_norm = values / max_abs
        elif normalize_method == 'minmax':
            # Normalize between 0 and 1 with protection against division by zero
            val_min = values.min()
            val_max = values.max()
            val_range = val_max - val_min
            
            if val_range == 0:
                # If min == max, assign constant value at center (0.5)
                print(f"Warning: Segment with constant values ({val_min:.3f}), assigning 0.5")
                values_norm = np.full_like(values, 0.5)
            else:
                values_norm = (values - val_min) / val_range
        else:
            values_norm = values
        
        # Verify that normalization was successful
        if np.any(np.isnan(values_norm)) or np.any(np.isinf(values_norm)):
            print(f"Warning: NaN/Inf values after normalization, skipping segment...")
            continue
        
        # Interpolate to common grid
        if len(time_norm) >= 2:
            try:
                interp_func = interp1d(time_norm, values_norm, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
                
                # Common grid from 0 to 1
                common_time = np.linspace(0, 1, num_points)
                interpolated_values = interp_func(common_time)
                
                # Verify interpolation
                if np.any(np.isnan(interpolated_values)) or np.any(np.isinf(interpolated_values)):
                    print(f"Warning: Interpolation failed, skipping segment...")
                    continue
                
                normalized_patterns.append(interpolated_values)
            except Exception as e:
                print(f"Error in interpolation: {e}, skipping segment...")
                continue
    
    if not normalized_patterns:
        print("Warning: Could not normalize any segment")
        return None
    
    # Convert to numpy array
    patterns_array = np.array(normalized_patterns)
    
    # Calculate statistics
    mean_pattern = np.mean(patterns_array, axis=0)
    variance_pattern = np.var(patterns_array, axis=0)
    std_pattern = np.std(patterns_array, axis=0)
    
    return {
        'time_normalized': np.linspace(0, 1, num_points),
        'mean_pattern': mean_pattern,
        'variance_pattern': variance_pattern,
        'std_pattern': std_pattern,
        'num_segments': len(normalized_patterns),
        'individual_patterns': patterns_array  # For additional analysis
    }

def plot_averaged_pattern(pattern_data, title="Average Pattern", n_std_levels=50):
    """
    Plots the average pattern with continuous color gradation for variance.
    """
    if pattern_data is None:
        print("No data to plot")
        return
    
    time_norm = pattern_data['time_normalized']
    mean = pattern_data['mean_pattern']
    std = pattern_data['std_pattern']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create color gradation for different standard deviation levels
    max_std_levels = 3  # Maximum number of standard deviations to show
    std_levels = np.linspace(0, max_std_levels, n_std_levels)
    
    # Create custom colormap: white in center, blue -> green -> yellow -> red at extremes
    colors_list = ['white', 'lightblue', 'skyblue', 'steelblue', 'blue', 'green', 'yellow', 'orange', 'red']
    n_colors = len(colors_list)
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors_list, N=n_std_levels)
    
    # Draw variance bands with gradation (from outside to inside)
    for i in reversed(range(1, len(std_levels))):  # Start from highest level
        std_level = std_levels[i]
        upper_bound = mean + std_level * std
        lower_bound = mean - std_level * std
        
        # Color should be more intense the greater the deviation
        color_intensity = std_level / max_std_levels
        color = custom_cmap(color_intensity)
        
        ax.fill_between(time_norm, lower_bound, upper_bound, 
                        color=color, alpha=0.7, linewidth=0)
    
    # Average pattern on top of everything
    ax.plot(time_norm, mean, 'black', linewidth=3, label='Average Pattern', zorder=10)
    
    # Add reference lines for 1σ and 2σ
    ax.plot(time_norm, mean + std, '--', color='darkred', alpha=0.8, linewidth=2, label='± 1σ', zorder=5)
    ax.plot(time_norm, mean - std, '--', color='darkred', alpha=0.8, linewidth=2, zorder=5)
    ax.plot(time_norm, mean + 2*std, '--', color='purple', alpha=0.8, linewidth=2, label='± 2σ', zorder=5)
    ax.plot(time_norm, mean - 2*std, '--', color='purple', alpha=0.8, linewidth=2, zorder=5)
    
    ax.set_xlabel('Normalized Time (0-1)')
    ax.set_ylabel('Normalized Value (0-1)')
    ax.set_title(f'{title} (n={pattern_data["num_segments"]} segments)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)
    
    # Add color bar to explain gradation (correctly oriented)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=max_std_levels))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Standard Deviations (σ)')
    
    plt.tight_layout()
    plt.savefig('averaged_pattern.png', dpi=300)
    plt.show()

# Improved alternative version with contours
def plot_averaged_pattern_contour_improved(pattern_data, title="Average Pattern"):
    """
    Improved version using contours with correct color gradation.
    """
    if pattern_data is None:
        print("No data to plot")
        return
    
    time_norm = pattern_data['time_normalized']
    mean = pattern_data['mean_pattern']
    std = pattern_data['std_pattern']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create denser mesh for contours
    X, Y = np.meshgrid(time_norm, np.linspace(0, 1.2, 200))
    
    # Calculate distance of each point to average curve in terms of standard deviations
    Z = np.zeros_like(X)
    for i, t in enumerate(time_norm):
        mean_val = mean[i]
        std_val = std[i] if std[i] > 0 else 0.01
        Z[:, i] = np.abs(Y[:, i] - mean_val) / std_val
    
    # Create custom colormap
    colors_list = ['white', 'lightblue', 'skyblue', 'steelblue', 'blue', 'green', 'yellow', 'orange', 'red']
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors_list, N=256)
    
    # Create smooth contours
    levels = np.linspace(0, 3, 30)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap=custom_cmap, alpha=0.8)
    
    # Average pattern
    ax.plot(time_norm, mean, 'black', linewidth=4, label='Average Pattern', zorder=10)
    
    # More visible reference lines
    ax.plot(time_norm, mean + std, '--', color='darkred', alpha=0.9, linewidth=2.5, label='± 1σ', zorder=5)
    ax.plot(time_norm, mean - std, '--', color='darkred', alpha=0.9, linewidth=2.5, zorder=5)
    ax.plot(time_norm, mean + 2*std, '--', color='purple', alpha=0.9, linewidth=2.5, label='± 2σ', zorder=5)
    ax.plot(time_norm, mean - 2*std, '--', color='purple', alpha=0.9, linewidth=2.5, zorder=5)
    
    ax.set_xlabel('Normalized Time (0-1)')
    ax.set_ylabel('Normalized Value (0-1)')
    ax.set_title(f'{title} (n={pattern_data["num_segments"]} segments)')
    ax.legend()
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_ylim(0, 1.2)
    
    # Color bar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Standard Deviations (σ)')
    
    plt.tight_layout()
    plt.savefig('averaged_pattern_contour_improved.png', dpi=300)
    plt.show()

def compare_segment_to_pattern(segment, pattern_data, feature='rssi', normalize_method='minmax'):
    """
    Compares an individual segment with the average pattern.
    Returns a similarity metric.
    """
    if pattern_data is None or len(segment['timestamp']) < 2:
        return None
    
    # Normalize segment in the same way
    time = segment['timestamp']
    values = segment[feature]
    
    # Temporal normalization with protection
    time_range = time.max() - time.min()
    if time_range == 0:
        print("Warning: Segment with constant time in comparison")
        return None
    time_norm = (time - time.min()) / time_range
    
    # Value normalization with protection
    if normalize_method == 'max':
        if feature == 'rssi':
            max_val = np.max(values)
            if max_val == 0:
                values_norm = values / (np.min(np.abs(values)) + 1e-8)
            else:
                values_norm = values / max_val
        else:
            max_abs = np.max(np.abs(values))
            if max_abs == 0:
                values_norm = np.zeros_like(values)
            else:
                values_norm = values / max_abs
    elif normalize_method == 'minmax':
        val_min = values.min()
        val_max = values.max()
        val_range = val_max - val_min
        
        if val_range == 0:
            values_norm = np.full_like(values, 0.5)
        else:
            values_norm = (values - val_min) / val_range
    else:
        values_norm = values
    
    # Verify normalization
    if np.any(np.isnan(values_norm)) or np.any(np.isinf(values_norm)):
        print("Warning: Normalization failed in comparison")
        return None
    
    # Interpolate to common grid
    try:
        interp_func = interp1d(time_norm, values_norm, kind='linear', 
                             bounds_error=False, fill_value='extrapolate')
        
        common_time = pattern_data['time_normalized']
        interpolated_values = interp_func(common_time)
        
        # Verify interpolation
        if np.any(np.isnan(interpolated_values)) or np.any(np.isinf(interpolated_values)):
            print("Warning: Interpolation failed in comparison")
            return None
        
    except Exception as e:
        print(f"Error in interpolation during comparison: {e}")
        return None
    
    # Calculate similarity metrics
    mean_pattern = pattern_data['mean_pattern']
    
    # Correlation
    try:
        correlation = np.corrcoef(interpolated_values, mean_pattern)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0  # No correlation if there are problems
    except:
        correlation = 0.0
    
    # Mean squared error
    mse = np.mean((interpolated_values - mean_pattern)**2)
    
    # Distance within variance
    std_pattern = pattern_data['std_pattern']
    z_scores = np.abs(interpolated_values - mean_pattern) / (std_pattern + 1e-8)
    mean_z_score = np.mean(z_scores)
    
    return {
        'correlation': correlation,
        'mse': mse,
        'mean_z_score': mean_z_score,
        'interpolated_values': interpolated_values
    }

# Example usage with your data
if __name__ == "__main__":

    # Load correct pass data

    import sys, os

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add src directory to path

    from csv_data_loader import extract_tag_data

    tags_data = extract_tag_data('data/dynamic.csv')
    tag_data = tags_data[list(tags_data.keys())[0]]
    
    # Segment into individual passes
    segments = split_tag_data_by_absolute_and_stat(tag_data, abs_threshold=1.0, stat_threshold=1.0)
    
    # Calculate average pattern for RSSI
    rssi_pattern = compute_averaged_pattern(segments, feature='rssi', num_points=100)
    
    # Plot
    plot_averaged_pattern(rssi_pattern, "Average RSSI Pattern")
    
    # Compare individual segment
    if segments:
        reference_segment = segments[5]
        similarity = compare_segment_to_pattern(reference_segment, rssi_pattern)
        print(f"Correlation: {similarity['correlation']:.3f}")
        print(f"MSE: {similarity['mse']:.3f}")
        print(f"Average Z-score: {similarity['mean_z_score']:.3f}")
        
        # Comparative plot of average pattern vs reference segment
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Subplot 1: Average pattern with variance bands
        time_norm = rssi_pattern['time_normalized']
        mean = rssi_pattern['mean_pattern']
        std = rssi_pattern['std_pattern']
        
        ax1.plot(time_norm, mean, 'blue', linewidth=3, label='Average Pattern')
        ax1.fill_between(time_norm, mean - std, mean + std, alpha=0.3, color='blue', label='± 1σ')
        ax1.fill_between(time_norm, mean - 2*std, mean + 2*std, alpha=0.2, color='blue', label='± 2σ')
        ax1.set_xlabel('Normalized Time (0-1)')
        ax1.set_ylabel('Normalized RSSI (0-1)')
        ax1.set_title(f'Average Pattern (n={rssi_pattern["num_segments"]} segments)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.2)
        
        # Subplot 2: Direct comparison
        # Normalize reference segment for visualization
        ref_time = reference_segment['timestamp']
        ref_rssi = reference_segment['rssi']
        
        # Temporal normalization
        ref_time_norm = (ref_time - ref_time.min()) / (ref_time.max() - ref_time.min())
        
        # RSSI minmax normalization
        ref_rssi_norm = (ref_rssi - ref_rssi.min()) / (ref_rssi.max() - ref_rssi.min())
        
        # Plot both in same subplot
        ax2.plot(time_norm, mean, 'blue', linewidth=3, label='Average Pattern', alpha=0.8)
        ax2.plot(ref_time_norm, ref_rssi_norm, 'red', linewidth=2, marker='o', 
                markersize=3, label=f'Segment #{1} (Duration: {reference_segment["duration"]:.2f}s)')
        
        # Show interpolated segment for comparison
        if similarity and 'interpolated_values' in similarity:
            ax2.plot(time_norm, similarity['interpolated_values'], 'orange', 
                    linewidth=2, linestyle='--', alpha=0.7, 
                    label='Interpolated Segment')
        
        # Add variance bands as reference
        ax2.fill_between(time_norm, mean - std, mean + std, alpha=0.2, color='blue')
        
        ax2.set_xlabel('Normalized Time (0-1)')
        ax2.set_ylabel('Normalized RSSI (0-1)')
        ax2.set_title(f'Comparison: Pattern vs Reference Segment\n'
                     f'Correlation: {similarity["correlation"]:.3f}, '
                     f'MSE: {similarity["mse"]:.3f}, '
                     f'Z-score: {similarity["mean_z_score"]:.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig('comparison_pattern_segment.png', dpi=300)
        plt.show()

    # Number of segments with average z-score less than 1, 2, 3
    if rssi_pattern:
        count_below_1 = 0
        count_below_2 = 0
        count_below_3 = 0
        
        for i, seg in enumerate(segments):
            sim = compare_segment_to_pattern(seg, rssi_pattern)
            if sim:
                if sim['mean_z_score'] < 1:
                    count_below_1 += 1
                if sim['mean_z_score'] < 2:
                    count_below_2 += 1
                if sim['mean_z_score'] < 3:
                    count_below_3 += 1
        
        print(f"Segments with average z-score < 1: {count_below_1}/{len(segments)}")
        print(f"Segments with average z-score < 2: {count_below_2}/{len(segments)}")
        print(f"Segments with average z-score < 3: {count_below_3}/{len(segments)}")