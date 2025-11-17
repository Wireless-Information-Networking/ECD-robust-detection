#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: interpolation.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Advanced interpolation and smoothing utilities for RFID signal processing.
    Provides multiple interpolation methods (linear, cubic, spline) with various
    smoothing techniques (Gaussian, Savitzky-Golay) for signal enhancement and
    arc detection preparation. Includes duplicate removal and quality analysis.

License: MIT License
Dependencies: numpy, matplotlib, scipy, read_csv (local)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline, splrep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
import sys
import os



def remove_duplicate_timestamps(timestamps, rssi, phase):
    """
    Removes duplicate timestamps maintaining the average of RSSI and phase values.
    
    :param timestamps: Array of timestamps
    :param rssi: Array of RSSI values
    :param phase: Array of phase values
    :return: Cleaned arrays without duplicates
    """
    # Find unique indices
    _, unique_indices = np.unique(timestamps, return_index=True)
    
    if len(unique_indices) == len(timestamps):
        # No duplicates
        return timestamps, rssi, phase
    
    # print(f"Found {len(timestamps) - len(unique_indices)} duplicate timestamps. Removing...")
    
    # Create unique arrays
    unique_timestamps = timestamps[unique_indices]
    unique_rssi = rssi[unique_indices]
    unique_phase = phase[unique_indices]
    
    # Sort by timestamp to ensure correct order
    sort_indices = np.argsort(unique_timestamps)
    
    return (unique_timestamps[sort_indices], 
            unique_rssi[sort_indices], 
            unique_phase[sort_indices])

def interpolate_and_smooth_segment(segment, num_points=200, kind='linear', 
                                 smoothing_method='gaussian', smoothing_params=None):
    """
    Interpolates and smooths an individual segment using advanced methods from interpolate_tag_data.
    
    :param segment: Dictionary with 'timestamp', 'rssi', 'phase'
    :param num_points: Number of points for interpolation
    :param kind: Type of interpolation ('linear', 'cubic', 'quadratic', 'spline')
    :param smoothing_method: Smoothing method ('gaussian', 'savgol', 'spline_smooth', None)
    :param smoothing_params: Specific parameters for smoothing method
    :return: Dictionary with interpolated and smoothed data
    """
    timestamps = segment['timestamp']
    rssi = segment['rssi']
    phase = segment['phase']
    
    # Verify we have sufficient points
    if len(timestamps) < 3:
        return {
            'timestamp': timestamps,
            'rssi': rssi,
            'phase': phase,
            'rssi_smooth': rssi,
            'phase_smooth': phase,
            'original_segment': segment,
            'interpolation_method': 'none',
            'smoothing_method': 'none'
        }
    
    # Remove duplicate timestamps
    timestamps_clean, rssi_clean, phase_clean = remove_duplicate_timestamps(timestamps, rssi, phase)
    
    # Create uniform temporal grid
    t_min = timestamps_clean.min()
    t_max = timestamps_clean.max()
    t_uniform = np.linspace(t_min, t_max, num_points)
    
    # Configure default smoothing parameters
    if smoothing_params is None:
        smoothing_params = {}
        
    # Set default smoothing parameters based on method
    if smoothing_method == 'gaussian':
        smoothing_params.setdefault('sigma', 1.0)  # Default for segments
    elif smoothing_method == 'savgol':
        smoothing_params.setdefault('window_length', min(21, num_points//4))
        smoothing_params.setdefault('polyorder', 3)
    elif smoothing_method == 'spline_smooth':
        smoothing_params.setdefault('smoothing_factor', len(timestamps_clean) * 0.1)
    
    try:
        if kind == 'spline' or smoothing_method == 'spline_smooth':
            # Use splines with controlled smoothing
            smoothing_factor = smoothing_params.get('smoothing_factor', len(timestamps_clean) * 0.1)
            
            # Spline for RSSI
            spline_rssi = UnivariateSpline(timestamps_clean, rssi_clean, s=smoothing_factor)
            rssi_interpolated = spline_rssi(t_uniform)
            
            # Spline for Phase
            spline_phase = UnivariateSpline(timestamps_clean, phase_clean, s=smoothing_factor)
            phase_interpolated = spline_phase(t_uniform)
            
            # For splines, the smoothing is already applied
            rssi_smooth = rssi_interpolated
            phase_smooth = phase_interpolated
            
        else:
            # Standard interpolation
            min_points_needed = {'linear': 2, 'quadratic': 3, 'cubic': 4}
            if len(timestamps_clean) < min_points_needed.get(kind, 2):
                print(f"Warning: Not enough unique points ({len(timestamps_clean)}) for {kind} interpolation. Using linear.")
                kind = 'linear'
            
            # Interpolate RSSI
            interp_func_rssi = interp1d(timestamps_clean, rssi_clean, kind=kind, 
                                       bounds_error=False, fill_value='extrapolate')
            rssi_interpolated = interp_func_rssi(t_uniform)
            
            # Interpolate Phase
            interp_func_phase = interp1d(timestamps_clean, phase_clean, kind=kind,
                                        bounds_error=False, fill_value='extrapolate')
            phase_interpolated = interp_func_phase(t_uniform)
            
            # Apply post-interpolation smoothing
            if smoothing_method == 'gaussian':
                sigma = smoothing_params.get('sigma', 1.0)
                rssi_smooth = gaussian_filter1d(rssi_interpolated, sigma=sigma)
                phase_smooth = gaussian_filter1d(phase_interpolated, sigma=sigma)
                
            elif smoothing_method == 'savgol':
                window_length = smoothing_params.get('window_length', min(21, len(rssi_interpolated)//4))
                # Ensure window_length is odd and valid
                if window_length % 2 == 0:
                    window_length += 1
                window_length = max(5, min(window_length, len(rssi_interpolated)))
                
                polyorder = smoothing_params.get('polyorder', 3)
                polyorder = min(polyorder, window_length - 1)
                
                rssi_smooth = savgol_filter(rssi_interpolated, window_length, polyorder)
                phase_smooth = savgol_filter(phase_interpolated, window_length, polyorder)
                
            elif smoothing_method is None:
                # No smoothing
                rssi_smooth = rssi_interpolated
                phase_smooth = phase_interpolated
                
            else:
                print(f"Unknown smoothing method: {smoothing_method}. No smoothing applied.")
                rssi_smooth = rssi_interpolated
                phase_smooth = phase_interpolated
        
        return {
            'timestamp': t_uniform,
            'rssi': rssi_interpolated,
            'phase': phase_interpolated,
            'rssi_smooth': rssi_smooth,
            'phase_smooth': phase_smooth,
            'original_segment': segment,
            'original_timestamps': timestamps,
            'original_rssi': rssi,
            'original_phase': phase,
            'clean_timestamps': timestamps_clean,
            'clean_rssi': rssi_clean,
            'clean_phase': phase_clean,
            'interpolation_method': kind,
            'smoothing_method': smoothing_method,
            'smoothing_params': smoothing_params
        }
        
    except Exception as e:
        print(f"Error in {kind} interpolation with {smoothing_method} smoothing: {e}")
        print("Fallback to linear interpolation with gaussian smoothing...")
        
        try:
            # Fallback to simple linear interpolation with basic gaussian smoothing
            interp_func_rssi = interp1d(timestamps_clean, rssi_clean, kind='linear', 
                                       bounds_error=False, fill_value='extrapolate')
            rssi_interpolated = interp_func_rssi(t_uniform)
            
            interp_func_phase = interp1d(timestamps_clean, phase_clean, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')
            phase_interpolated = interp_func_phase(t_uniform)
            
            # Apply basic gaussian smoothing
            rssi_smooth = gaussian_filter1d(rssi_interpolated, sigma=1.0)
            phase_smooth = gaussian_filter1d(phase_interpolated, sigma=1.0)
            
            return {
                'timestamp': t_uniform,
                'rssi': rssi_interpolated,
                'phase': phase_interpolated,
                'rssi_smooth': rssi_smooth,
                'phase_smooth': phase_smooth,
                'original_segment': segment,
                'interpolation_method': 'linear_fallback',
                'smoothing_method': 'gaussian_fallback'
            }
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            # Return original data if all else fails
            return {
                'timestamp': timestamps,
                'rssi': rssi,
                'phase': phase,
                'rssi_smooth': rssi,
                'phase_smooth': phase,
                'original_segment': segment,
                'interpolation_method': 'none',
                'smoothing_method': 'none'
            }

def detect_local_minima(signal, min_distance=10, prominence_threshold=0.1):
    """
    Detects local minima in a smoothed signal for arc segmentation.
    
    :param signal: Smoothed signal (RSSI)
    :param min_distance: Minimum distance between minima
    :param prominence_threshold: Prominence threshold for minima
    :return: Indices of local minima
    """
    # Invert the signal to use find_peaks (find maxima in inverted signal = minima in original)
    inverted_signal = -signal
    
    # Normalize to calculate prominence
    signal_norm = (signal - signal.min()) / (signal.max() - signal.min()) if signal.max() != signal.min() else signal
    prominence_abs = prominence_threshold * (signal.max() - signal.min())
    
    # Find peaks in inverted signal (minima in original)
    peaks, properties = find_peaks(inverted_signal, 
                                  distance=min_distance,
                                  prominence=prominence_abs)
    
    return peaks

def interpolate_tag_data(tag_values, num_points=1000, kind='linear', smoothing_method=None, smoothing_params=None):
    """
    Interpolates tag data to a uniform temporal grid with smoothing options.
    
    :param tag_values: Dictionary with 'timestamp', 'rssi', 'phase'
    :param num_points: Number of points for interpolation
    :param kind: Type of interpolation ('linear', 'cubic', 'quadratic', 'spline')
    :param smoothing_method: Smoothing method ('gaussian', 'savgol', 'spline_smooth', None)
    :param smoothing_params: Specific parameters for smoothing method
    :return: Dictionary with interpolated data
    """
    timestamps = tag_values['timestamp']
    rssi = tag_values['rssi']
    phase = tag_values['phase']
    
    # Remove duplicate timestamps
    timestamps_clean, rssi_clean, phase_clean = remove_duplicate_timestamps(timestamps, rssi, phase)
    
    # Create uniform temporal grid
    t_min = timestamps_clean.min()
    t_max = timestamps_clean.max()
    t_uniform = np.linspace(t_min, t_max, num_points)
    
    # Configure default smoothing parameters
    if smoothing_params is None:
        smoothing_params = {}
    
    try:
        if kind == 'spline' or smoothing_method == 'spline_smooth':
            # Use splines with controlled smoothing
            smoothing_factor = smoothing_params.get('smoothing_factor', len(timestamps_clean) * 0.1)
            
            # Spline for RSSI
            spline_rssi = UnivariateSpline(timestamps_clean, rssi_clean, s=smoothing_factor)
            rssi_interpolated = spline_rssi(t_uniform)
            
            # Spline for Phase
            spline_phase = UnivariateSpline(timestamps_clean, phase_clean, s=smoothing_factor)
            phase_interpolated = spline_phase(t_uniform)
            
        else:
            # Standard interpolation
            min_points_needed = {'linear': 2, 'quadratic': 3, 'cubic': 4}
            if len(timestamps_clean) < min_points_needed.get(kind, 2):
                print(f"Warning: Not enough unique points ({len(timestamps_clean)}) for {kind} interpolation. Using linear.")
                kind = 'linear'
            
            # Interpolate RSSI
            interp_func_rssi = interp1d(timestamps_clean, rssi_clean, kind=kind, 
                                       bounds_error=False, fill_value='extrapolate')
            rssi_interpolated = interp_func_rssi(t_uniform)
            
            # Interpolate Phase
            interp_func_phase = interp1d(timestamps_clean, phase_clean, kind=kind,
                                        bounds_error=False, fill_value='extrapolate')
            phase_interpolated = interp_func_phase(t_uniform)
        
        # Apply post-interpolation smoothing
        if smoothing_method == 'gaussian':
            sigma = smoothing_params.get('sigma', 2.0)
            rssi_interpolated = gaussian_filter1d(rssi_interpolated, sigma=sigma)
            phase_interpolated = gaussian_filter1d(phase_interpolated, sigma=sigma)
            
        elif smoothing_method == 'savgol':
            window_length = smoothing_params.get('window_length', min(51, len(rssi_interpolated)//4))
            # Ensure window_length is odd and valid
            if window_length % 2 == 0:
                window_length += 1
            window_length = max(5, min(window_length, len(rssi_interpolated)))
            
            polyorder = smoothing_params.get('polyorder', 3)
            polyorder = min(polyorder, window_length - 1)
            
            rssi_interpolated = savgol_filter(rssi_interpolated, window_length, polyorder)
            phase_interpolated = savgol_filter(phase_interpolated, window_length, polyorder)
            
    except Exception as e:
        print(f"Error in {kind} interpolation with {smoothing_method} smoothing: {e}")
        print("Fallback to linear interpolation without smoothing...")
        
        # Fallback to linear interpolation
        interp_func_rssi = interp1d(timestamps_clean, rssi_clean, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
        rssi_interpolated = interp_func_rssi(t_uniform)
        
        interp_func_phase = interp1d(timestamps_clean, phase_clean, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        phase_interpolated = interp_func_phase(t_uniform)
    
    return {
        'timestamp': t_uniform,
        'rssi': rssi_interpolated,
        'phase': phase_interpolated,
        'original_timestamps': timestamps,
        'original_rssi': rssi,
        'original_phase': phase,
        'clean_timestamps': timestamps_clean,
        'clean_rssi': rssi_clean,
        'clean_phase': phase_clean,
        'interpolation_method': kind,
        'smoothing_method': smoothing_method
    }

def plot_comparison(original_data, interpolated_data, tag_id):
    """
    Plots original vs interpolated data comparison.
    
    :param original_data: Original tag data
    :param interpolated_data: Interpolated tag data
    :param tag_id: Tag identifier for plot titles
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot RSSI
    ax1.scatter(interpolated_data['original_timestamps'], interpolated_data['original_rssi'], 
               color='blue', alpha=0.4, s=10, label='Original RSSI')
    if 'clean_timestamps' in interpolated_data:
        ax1.scatter(interpolated_data['clean_timestamps'], interpolated_data['clean_rssi'], 
                   color='cyan', alpha=0.8, s=15, label='Clean RSSI (no duplicates)')
    ax1.plot(interpolated_data['timestamp'], interpolated_data['rssi'], 
            color='red', linewidth=1.5, label='Interpolated RSSI')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('RSSI (dBm)')
    ax1.set_title(f'Tag {tag_id} - RSSI Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Phase
    ax2.scatter(interpolated_data['original_timestamps'], interpolated_data['original_phase'], 
               color='green', alpha=0.4, s=10, label='Original Phase')
    if 'clean_timestamps' in interpolated_data:
        ax2.scatter(interpolated_data['clean_timestamps'], interpolated_data['clean_phase'], 
                   color='lime', alpha=0.8, s=15, label='Clean Phase (no duplicates)')
    ax2.plot(interpolated_data['timestamp'], interpolated_data['phase'], 
            color='orange', linewidth=1.5, label='Interpolated Phase')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Phase')
    ax2.set_title(f'Tag {tag_id} - Phase Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output_plots', exist_ok=True)
    
    # Save figure
    plt.savefig(f'output_plots/comparison_tag_{tag_id}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_interpolation_quality(original_data, interpolated_data, tag_id):
    """
    Plots interpolation quality by showing errors.
    
    :param original_data: Original tag data
    :param interpolated_data: Interpolated tag data  
    :param tag_id: Tag identifier
    :return: Dictionary with error metrics
    """
    # Use clean data for error calculation
    if 'clean_timestamps' in interpolated_data:
        ref_timestamps = interpolated_data['clean_timestamps']
        ref_rssi = interpolated_data['clean_rssi']
        ref_phase = interpolated_data['clean_phase']
    else:
        ref_timestamps = interpolated_data['original_timestamps']
        ref_rssi = interpolated_data['original_rssi']
        ref_phase = interpolated_data['original_phase']
    
    # Interpolate the interpolated data back to reference timestamps
    interp_func_rssi = interp1d(interpolated_data['timestamp'], interpolated_data['rssi'], 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_func_phase = interp1d(interpolated_data['timestamp'], interpolated_data['phase'], 
                                kind='linear', bounds_error=False, fill_value='extrapolate')
    
    rssi_back = interp_func_rssi(ref_timestamps)
    phase_back = interp_func_phase(ref_timestamps)
    
    # Calculate errors
    rssi_error = np.abs(ref_rssi - rssi_back)
    phase_error = np.abs(ref_phase - phase_back)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # RSSI Error
    ax1.plot(ref_timestamps, rssi_error, 'r-', linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Absolute RSSI Error (dBm)')
    ax1.set_title(f'Tag {tag_id} - RSSI Interpolation Error\nMean error: {np.mean(rssi_error):.4f} dBm')
    ax1.grid(True, alpha=0.3)
    
    # Phase Error
    ax2.plot(ref_timestamps, phase_error, 'b-', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Absolute Phase Error')
    ax2.set_title(f'Tag {tag_id} - Phase Interpolation Error\nMean error: {np.mean(phase_error):.4f}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output_plots', exist_ok=True)
    
    # Save figure
    plt.savefig(f'output_plots/interpolation_quality_tag_{tag_id}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'rssi_mean_error': np.mean(rssi_error),
        'rssi_max_error': np.max(rssi_error),
        'phase_mean_error': np.mean(phase_error),
        'phase_max_error': np.max(phase_error)
    }

def compare_interpolation_methods(tag_values, tag_id, num_points=1000):
    """
    Compares different interpolation methods with robust error handling.
    
    :param tag_values: Tag data dictionary
    :param tag_id: Tag identifier
    :param num_points: Number of interpolation points
    """
    methods = ['linear', 'quadratic', 'cubic']
    successful_methods = []
    
    # First check which methods work
    for method in methods:
        try:
            test_interpolated = interpolate_tag_data(tag_values, 10, kind=method)  # Quick test
            successful_methods.append(method)
        except Exception as e:
            print(f"Method {method} not available: {e}")
    
    if not successful_methods:
        print("No interpolation methods available.")
        return
    
    fig, axes = plt.subplots(len(successful_methods), 2, figsize=(15, 4*len(successful_methods)))
    
    # If only one method, axes won't be a 2D list
    if len(successful_methods) == 1:
        axes = [axes]
    
    for i, method in enumerate(successful_methods):
        interpolated = interpolate_tag_data(tag_values, num_points, kind=method)
        
        # RSSI
        axes[i][0].scatter(tag_values['timestamp'], tag_values['rssi'], 
                          color='blue', alpha=0.6, s=15, label='Original')
        axes[i][0].plot(interpolated['timestamp'], interpolated['rssi'], 
                       color='red', linewidth=1, label=f'{method.capitalize()}')
        axes[i][0].set_ylabel('RSSI (dBm)')
        axes[i][0].set_title(f'Tag {tag_id} - RSSI ({method})')
        axes[i][0].legend()
        axes[i][0].grid(True, alpha=0.3)
        
        # Phase
        axes[i][1].scatter(tag_values['timestamp'], tag_values['phase'], 
                          color='green', alpha=0.6, s=15, label='Original')
        axes[i][1].plot(interpolated['timestamp'], interpolated['phase'], 
                       color='orange', linewidth=1, label=f'{method.capitalize()}')
        axes[i][1].set_ylabel('Phase')
        axes[i][1].set_title(f'Tag {tag_id} - Phase ({method})')
        axes[i][1].legend()
        axes[i][1].grid(True, alpha=0.3)
        
        if i == len(successful_methods) - 1:
            axes[i][0].set_xlabel('Time (s)')
            axes[i][1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output_plots', exist_ok=True)
    
    # Save figure
    plt.savefig(f'output_plots/comparison_methods_tag_{tag_id}.png', dpi=300, bbox_inches='tight')
    plt.show()

def interpolate_multiple_tags(tag_data_dict, num_points=1000, kind='linear'):
    """
    Interpolates multiple tags and returns a dictionary with all results.
    
    :param tag_data_dict: Dictionary with multiple tag data
    :param num_points: Number of interpolation points
    :param kind: Interpolation method
    :return: Dictionary with interpolated results
    """
    interpolated_dict = {}
    
    for tag_id, tag_values in tag_data_dict.items():
        try:
            interpolated_dict[tag_id] = interpolate_tag_data(tag_values, num_points, kind)
            print(f"Tag {tag_id}: Interpolation successful")
        except Exception as e:
            print(f"Error interpolating tag {tag_id}: {e}")
    
    return interpolated_dict

def compare_smoothing_methods(tag_values, tag_id, num_points=1000):
    """
    Compares different smoothing methods in interpolation.
    
    :param tag_values: Tag data dictionary
    :param tag_id: Tag identifier
    :param num_points: Number of interpolation points
    :return: List of interpolated results
    """
    # Define methods to compare
    methods = [
        {'kind': 'linear', 'smoothing_method': None, 'name': 'Linear without smoothing'},
        {'kind': 'cubic', 'smoothing_method': None, 'name': 'Cubic without smoothing'},
        {'kind': 'linear', 'smoothing_method': 'gaussian', 'smoothing_params': {'sigma': 1.0}, 'name': 'Linear + Gaussian (σ=1)'},
        {'kind': 'linear', 'smoothing_method': 'gaussian', 'smoothing_params': {'sigma': 3.0}, 'name': 'Linear + Gaussian (σ=3)'},
        {'kind': 'linear', 'smoothing_method': 'savgol', 'smoothing_params': {'window_length': 21, 'polyorder': 3}, 'name': 'Linear + Savitzky-Golay'},
        {'kind': 'spline', 'smoothing_params': {'smoothing_factor': len(tag_values['timestamp']) * 0.1}, 'name': 'Smooth spline'}
    ]
    
    successful_methods = []
    interpolated_results = []
    
    # Test each method
    for method in methods:
        try:
            interpolated = interpolate_tag_data(
                tag_values, 
                num_points, 
                kind=method['kind'],
                smoothing_method=method.get('smoothing_method'),
                smoothing_params=method.get('smoothing_params')
            )
            successful_methods.append(method['name'])
            interpolated_results.append(interpolated)
        except Exception as e:
            print(f"Error with method {method['name']}: {e}")
    
    if not successful_methods:
        print("No smoothing methods available.")
        return
    
    # Create visualization
    fig, axes = plt.subplots(len(successful_methods), 2, figsize=(18, 4*len(successful_methods)))
    
    if len(successful_methods) == 1:
        axes = [axes]
    
    for i, (method_name, interpolated) in enumerate(zip(successful_methods, interpolated_results)):
        # RSSI
        axes[i][0].scatter(tag_values['timestamp'], tag_values['rssi'], 
                          color='blue', alpha=0.4, s=8, label='Original', zorder=1)
        axes[i][0].plot(interpolated['timestamp'], interpolated['rssi'], 
                       color='red', linewidth=2, label='Interpolated', zorder=2)
        axes[i][0].set_ylabel('RSSI (dBm)')
        axes[i][0].set_title(f'{method_name} - RSSI')
        axes[i][0].legend()
        axes[i][0].grid(True, alpha=0.3)
        
        # Phase
        axes[i][1].scatter(tag_values['timestamp'], tag_values['phase'], 
                          color='green', alpha=0.4, s=8, label='Original', zorder=1)
        axes[i][1].plot(interpolated['timestamp'], interpolated['phase'], 
                       color='orange', linewidth=2, label='Interpolated', zorder=2)
        axes[i][1].set_ylabel('Phase')
        axes[i][1].set_title(f'{method_name} - Phase')
        axes[i][1].legend()
        axes[i][1].grid(True, alpha=0.3)
        
        if i == len(successful_methods) - 1:
            axes[i][0].set_xlabel('Time (s)')
            axes[i][1].set_xlabel('Time (s)')
    
    plt.suptitle(f'Tag {tag_id} - Smoothing Methods Comparison', fontsize=16)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output_plots', exist_ok=True)
    
    # Save figure
    plt.savefig(f'output_plots/smoothing_comparison_tag_{tag_id}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return interpolated_results

def plot_smoothing_analysis(tag_values, tag_id, num_points=1000):
    """
    Detailed analysis of smoothing effects on data.
    
    :param tag_values: Tag data dictionary
    :param tag_id: Tag identifier
    :param num_points: Number of interpolation points
    """
    # Create interpolations with different smoothing levels
    smoothing_levels = [
        {'sigma': 0.5, 'name': 'Light (σ=0.5)'},
        {'sigma': 1.0, 'name': 'Moderate (σ=1.0)'},
        {'sigma': 2.0, 'name': 'Strong (σ=2.0)'},
        {'sigma': 4.0, 'name': 'Very strong (σ=4.0)'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot original
    axes[0,0].scatter(tag_values['timestamp'], tag_values['rssi'], 
                     color='blue', alpha=0.5, s=10, label='Original')
    axes[0,1].scatter(tag_values['timestamp'], tag_values['phase'], 
                     color='green', alpha=0.5, s=10, label='Original')
    
    colors = ['red', 'orange', 'purple', 'brown']
    
    for i, (level, color) in enumerate(zip(smoothing_levels, colors)):
        interpolated = interpolate_tag_data(
            tag_values, 
            num_points, 
            kind='cubic',
            smoothing_method='gaussian',
            smoothing_params={'sigma': level['sigma']}
        )
        
        # RSSI
        axes[0,0].plot(interpolated['timestamp'], interpolated['rssi'], 
                      color=color, linewidth=2, label=level['name'])
        
        # Phase
        axes[0,1].plot(interpolated['timestamp'], interpolated['phase'], 
                      color=color, linewidth=2, label=level['name'])
    
    axes[0,0].set_title('RSSI - Different Gaussian smoothing levels')
    axes[0,0].set_ylabel('RSSI (dBm)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].set_title('Phase - Different Gaussian smoothing levels')
    axes[0,1].set_ylabel('Phase')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Compare Savitzky-Golay with different windows
    savgol_windows = [11, 21, 51, 101]
    axes[1,0].scatter(tag_values['timestamp'], tag_values['rssi'], 
                     color='blue', alpha=0.5, s=10, label='Original')
    
    for i, (window, color) in enumerate(zip(savgol_windows, colors)):
        try:
            interpolated = interpolate_tag_data(
                tag_values, 
                num_points, 
                kind='linear',
                smoothing_method='savgol',
                smoothing_params={'window_length': window, 'polyorder': 3}
            )
            
            axes[1,0].plot(interpolated['timestamp'], interpolated['rssi'], 
                          color=color, linewidth=2, label=f'Savgol (window={window})')
        except Exception as e:
            print(f"Error with window {window}: {e}")
    
    axes[1,0].set_title('RSSI - Savitzky-Golay with different windows')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('RSSI (dBm)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Compare splines with different smoothing factors
    spline_factors = [0.01, 0.1, 1.0, 10.0]
    n_points = len(tag_values['timestamp'])
    axes[1,1].scatter(tag_values['timestamp'], tag_values['rssi'], 
                     color='blue', alpha=0.5, s=10, label='Original')
    
    for i, (factor, color) in enumerate(zip(spline_factors, colors)):
        try:
            interpolated = interpolate_tag_data(
                tag_values, 
                num_points, 
                kind='spline',
                smoothing_params={'smoothing_factor': n_points * factor}
            )
            
            axes[1,1].plot(interpolated['timestamp'], interpolated['rssi'], 
                          color=color, linewidth=2, label=f'Spline (s={factor}×N)')
        except Exception as e:
            print(f"Error with factor {factor}: {e}")
    
    axes[1,1].set_title('RSSI - Splines with different smoothing factors')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('RSSI (dBm)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Tag {tag_id} - Smoothing Analysis', fontsize=16)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output_plots', exist_ok=True)
    
    # Save figure
    plt.savefig(f'output_plots/smoothing_analysis_tag_{tag_id}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    from csv_data_loader import extract_tag_data

    # Load data
    print("Loading data from data/test/0cmvert-1.csv...")
    tag_data = extract_tag_data('data/test/0cmvert-1.csv')
    
    if not tag_data:
        print("Could not load data.")
        exit()
    
    # Select first tag for example
    tag_id = list(tag_data.keys())[0]
    tag_values = tag_data[tag_id]

    # Convert lists to numpy arrays
    tag_values['timestamp'] = np.array(tag_values['timestamp'])
    tag_values['rssi'] = np.array(tag_values['rssi'])
    tag_values['phase'] = np.array(tag_values['phase'])
    
    # Reduce points to speed up processing
    max_points = 2000
    if len(tag_values['timestamp']) > max_points:
        print(f"Reducing points from {len(tag_values['timestamp'])} to {max_points}")
        tag_values['timestamp'] = tag_values['timestamp'][:max_points]
        tag_values['rssi'] = tag_values['rssi'][:max_points]
        tag_values['phase'] = tag_values['phase'][:max_points]
    
    print(f"Processing tag: {tag_id}")
    
    # Recommended smooth interpolation
    print("\n=== Recommended smooth interpolation ===")
    interpolated_smooth = interpolate_tag_data(
        tag_values, 
        num_points=1000, 
        kind='cubic',
        smoothing_method='gaussian',
        smoothing_params={'sigma': 2.0}
    )
    
    # Plot comparison
    plot_comparison(tag_values, interpolated_smooth, f"{tag_id}_smooth")
    
    # Compare smoothing methods
    print("\n=== Comparing smoothing methods ===")
    compare_smoothing_methods(tag_values, tag_id)
    
    # Detailed smoothing analysis
    print("\n=== Detailed smoothing analysis ===")
    plot_smoothing_analysis(tag_values, tag_id)
    
    print("\n=== Recommendations for smooth interpolation ===")
    print("1. For noisy data: use 'gaussian' with sigma=2.0-4.0")
    print("2. To preserve features: use 'savgol' with window_length=21")
    print("3. For maximum smoothness: use 'spline' with high smoothing_factor")
    print("4. For balance: use 'cubic' + 'gaussian' with sigma=1.0-2.0")