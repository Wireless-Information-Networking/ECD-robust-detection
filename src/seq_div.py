#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: seq_div.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Advanced RFID tag data segmentation script with hierarchical pass and arc detection.
    Implements multiple methods including peak detection, derivative analysis, thresholding,
    and DBSCAN clustering for robust segmentation of interpolated RFID signals into
    individual passes and arcs with comprehensive visualization and comparison tools.

License: MIT License
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from interpolation import interpolate_tag_data
import os

from csv_data_loader import extract_tag_data

def detect_passes_from_interpolated(interpolated_data, time_gap_threshold=2.0):
    """
    Detects independent passes in interpolated data based on temporal gaps.
    Adapts methods from timediff.py to work with interpolated data.
    
    :param interpolated_data: Dictionary with interpolated timestamp, rssi, phase
    :param time_gap_threshold: Minimum time gap to consider separate passes
    :return: List of detected passes
    """
    timestamps = interpolated_data['timestamp']
    rssi = interpolated_data['rssi']
    phase = interpolated_data['phase']
    
    # Calculate time differences
    time_diffs = np.diff(timestamps)
    
    # Detect significant gaps (between passes)
    # Use robust statistics to detect temporal outliers
    q1, q3 = np.percentile(time_diffs, [25, 75])
    iqr = q3 - q1
    gap_threshold = q3 + 1.5 * iqr  # Tukey method
    
    # If automatic threshold is too small, use specified minimum
    gap_threshold = max(gap_threshold, time_gap_threshold)
    
    # Find indices where there are large gaps (between passes)
    gap_indices = np.where(time_diffs > gap_threshold)[0] + 1
    
    # Create split points
    split_indices = [0] + gap_indices.tolist() + [len(timestamps)]
    
    passes = []
    for start_idx, end_idx in zip(split_indices[:-1], split_indices[1:]):
        if end_idx - start_idx > 10:  # Minimum points per pass
            duration = timestamps[end_idx-1] - timestamps[start_idx]
            passes.append({
                'timestamp': timestamps[start_idx:end_idx],
                'rssi': rssi[start_idx:end_idx],
                'phase': phase[start_idx:end_idx],
                'duration': duration,
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx-1],
                'num_points': end_idx - start_idx
            })
    
    print(f"Detected {len(passes)} passes with gaps > {gap_threshold:.3f}s")
    return passes

def _detect_arcs_by_peaks_in_pass(timestamps, rssi, phase, rssi_norm, min_prominence, min_width):
    """Detects arcs by peaks within a pass."""
    # Smooth slightly
    rssi_smooth = gaussian_filter1d(rssi_norm, sigma=0.5)
    
    # Detect peaks
    peaks, properties = find_peaks(rssi_smooth, 
                                  prominence=min_prominence,
                                  width=min_width,
                                  distance=min_width)
    
    if len(peaks) == 0:
        # No clear peaks, return entire pass as one arc
        return [{
            'timestamp': timestamps,
            'rssi': rssi,
            'phase': phase,
            'duration': timestamps[-1] - timestamps[0],
            'peak_idx': np.argmax(rssi),
            'peak_rssi': rssi[np.argmax(rssi)],
            'start_time': timestamps[0],
            'end_time': timestamps[-1]
        }]
    
    # Calculate arc boundaries based on peak widths
    widths_result = peak_widths(rssi_smooth, peaks, rel_height=0.3)
    left_bases = widths_result[2].astype(int)
    right_bases = widths_result[3].astype(int)
    
    arcs = []
    for i, peak_idx in enumerate(peaks):
        start_idx = max(0, left_bases[i])
        end_idx = min(len(timestamps), right_bases[i] + 1)
        
        if end_idx > start_idx:
            duration = timestamps[end_idx-1] - timestamps[start_idx]
            arcs.append({
                'timestamp': timestamps[start_idx:end_idx],
                'rssi': rssi[start_idx:end_idx],
                'phase': phase[start_idx:end_idx],
                'duration': duration,
                'peak_idx': peak_idx - start_idx,
                'peak_rssi': rssi[peak_idx],
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx-1]
            })
    
    return arcs

def _detect_arcs_by_derivative_in_pass(timestamps, rssi, phase, rssi_norm):
    """Detects arcs by derivative within a pass."""
    rssi_smooth = gaussian_filter1d(rssi_norm, sigma=1)
    derivative = np.gradient(rssi_smooth)
    
    # Find zero crossings
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    
    arcs = []
    for i in range(0, len(zero_crossings) - 1, 2):
        if i + 1 < len(zero_crossings):
            start_idx = zero_crossings[i]
            end_idx = min(zero_crossings[i + 1] + 1, len(timestamps))
            
            if end_idx > start_idx:
                segment_rssi = rssi[start_idx:end_idx]
                peak_idx_local = np.argmax(segment_rssi)
                duration = timestamps[end_idx-1] - timestamps[start_idx]
                
                arcs.append({
                    'timestamp': timestamps[start_idx:end_idx],
                    'rssi': rssi[start_idx:end_idx],
                    'phase': phase[start_idx:end_idx],
                    'duration': duration,
                    'peak_idx': peak_idx_local,
                    'peak_rssi': segment_rssi[peak_idx_local],
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx-1]
                })
    
    return arcs if arcs else [{
        'timestamp': timestamps,
        'rssi': rssi,
        'phase': phase,
        'duration': timestamps[-1] - timestamps[0],
        'peak_idx': np.argmax(rssi),
        'peak_rssi': rssi[np.argmax(rssi)],
        'start_time': timestamps[0],
        'end_time': timestamps[-1]
    }]

def _detect_arcs_by_threshold_in_pass(timestamps, rssi, phase, rssi_norm):
    """Detects arcs by threshold within a pass."""
    # Use lower percentile to detect multiple arcs within pass
    threshold = np.percentile(rssi, 30)
    above_threshold = rssi > threshold
    
    # Find continuous regions above threshold
    diff = np.diff(above_threshold.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    if above_threshold[0]:
        starts = np.insert(starts, 0, 0)
    if above_threshold[-1]:
        ends = np.append(ends, len(above_threshold))
    
    arcs = []
    for start_idx, end_idx in zip(starts, ends):
        if end_idx > start_idx:
            segment_rssi = rssi[start_idx:end_idx]
            peak_idx_local = np.argmax(segment_rssi)
            duration = timestamps[end_idx-1] - timestamps[start_idx]
            
            arcs.append({
                'timestamp': timestamps[start_idx:end_idx],
                'rssi': rssi[start_idx:end_idx],
                'phase': phase[start_idx:end_idx],
                'duration': duration,
                'peak_idx': peak_idx_local,
                'peak_rssi': segment_rssi[peak_idx_local],
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx-1]
            })
    
    return arcs if arcs else [{
        'timestamp': timestamps,
        'rssi': rssi,
        'phase': phase,
        'duration': timestamps[-1] - timestamps[0],
        'peak_idx': np.argmax(rssi),
        'peak_rssi': rssi[np.argmax(rssi)],
        'start_time': timestamps[0],
        'end_time': timestamps[-1]
    }]

def _detect_arcs_by_dbscan_in_pass(timestamps, rssi, phase, rssi_norm, eps=0.1, min_samples=5):
    """
    Detects arcs using DBSCAN in time-RSSI space within a pass.
    
    :param timestamps: Time array
    :param rssi: RSSI array
    :param phase: Phase array
    :param rssi_norm: Normalized RSSI array
    :param eps: DBSCAN epsilon parameter
    :param min_samples: DBSCAN minimum samples parameter
    :return: List of detected arcs
    """
    # Prepare data for clustering
    # Normalize time to have similar weight as RSSI
    time_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    
    # Create feature matrix: [normalized_time, normalized_rssi]
    features = np.column_stack([time_norm, rssi_norm])
    
    # Scale features for DBSCAN
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features_scaled)
    
    # Filter noise (label -1)
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels != -1]
    
    if len(unique_labels) == 0:
        # No clusters found, return entire pass as one arc
        return [{
            'timestamp': timestamps,
            'rssi': rssi,
            'phase': phase,
            'duration': timestamps[-1] - timestamps[0],
            'peak_idx': np.argmax(rssi),
            'peak_rssi': rssi[np.argmax(rssi)],
            'start_time': timestamps[0],
            'end_time': timestamps[-1],
            'cluster_id': 0,
            'noise_points': np.sum(cluster_labels == -1)
        }]
    
    arcs = []
    for label in unique_labels:
        cluster_mask = cluster_labels == label
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 0:
            # Ensure indices are contiguous to form valid arc
            start_idx = cluster_indices.min()
            end_idx = cluster_indices.max() + 1
            
            segment_rssi = rssi[start_idx:end_idx]
            peak_idx_local = np.argmax(segment_rssi)
            duration = timestamps[end_idx-1] - timestamps[start_idx]
            
            arcs.append({
                'timestamp': timestamps[start_idx:end_idx],
                'rssi': rssi[start_idx:end_idx],
                'phase': phase[start_idx:end_idx],
                'duration': duration,
                'peak_idx': peak_idx_local,
                'peak_rssi': segment_rssi[peak_idx_local],
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx-1],
                'cluster_id': int(label),
                'cluster_size': np.sum(cluster_mask),
                'noise_points': np.sum(cluster_labels == -1)
            })
    
    # Sort arcs by start time
    arcs.sort(key=lambda x: x['start_time'])
    
    return arcs

def _detect_arcs_by_dbscan_advanced_in_pass(timestamps, rssi, phase, rssi_norm, 
                                           use_phase=True, use_derivative=True):
    """
    Advanced DBSCAN version that includes more features.
    
    :param timestamps: Time array
    :param rssi: RSSI array
    :param phase: Phase array
    :param rssi_norm: Normalized RSSI array
    :param use_phase: Whether to include phase features
    :param use_derivative: Whether to include derivative features
    :return: List of detected arcs
    """
    # Basic features
    time_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    features = [time_norm, rssi_norm]
    feature_names = ['time', 'rssi']
    
    # Add phase if available
    if use_phase and len(phase) > 0:
        phase_norm = (phase - phase.min()) / (phase.max() - phase.min()) if phase.max() != phase.min() else np.zeros_like(phase)
        features.append(phase_norm)
        feature_names.append('phase')
    
    # Add derivatives as features
    if use_derivative:
        rssi_derivative = np.gradient(rssi_norm)
        rssi_derivative_norm = (rssi_derivative - rssi_derivative.min()) / (rssi_derivative.max() - rssi_derivative.min()) if rssi_derivative.max() != rssi_derivative.min() else np.zeros_like(rssi_derivative)
        features.append(rssi_derivative_norm)
        feature_names.append('rssi_derivative')
        
        if use_phase and len(phase) > 0:
            phase_derivative = np.gradient(phase_norm)
            phase_derivative_norm = (phase_derivative - phase_derivative.min()) / (phase_derivative.max() - phase_derivative.min()) if phase_derivative.max() != phase_derivative.min() else np.zeros_like(phase_derivative)
            features.append(phase_derivative_norm)
            feature_names.append('phase_derivative')
    
    # Create feature matrix
    features_matrix = np.column_stack(features)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_matrix)
    
    # Adaptive parameters for DBSCAN
    n_points = len(timestamps)
    eps = max(0.05, min(0.3, 1.0 / np.sqrt(n_points)))  # Adaptive eps
    min_samples = max(3, min(10, n_points // 20))        # Adaptive min_samples
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features_scaled)
    
    # Analyze results
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels != -1]
    noise_points = np.sum(cluster_labels == -1)
    
    print(f"  Advanced DBSCAN: {len(unique_labels)} clusters, {noise_points} noise points")
    print(f"  Features used: {feature_names}")
    print(f"  Parameters: eps={eps:.3f}, min_samples={min_samples}")
    
    if len(unique_labels) == 0:
        return [{
            'timestamp': timestamps,
            'rssi': rssi,
            'phase': phase,
            'duration': timestamps[-1] - timestamps[0],
            'peak_idx': np.argmax(rssi),
            'peak_rssi': rssi[np.argmax(rssi)],
            'start_time': timestamps[0],
            'end_time': timestamps[-1],
            'cluster_id': 0,
            'noise_points': noise_points,
            'method': 'dbscan_advanced'
        }]
    
    arcs = []
    for label in unique_labels:
        cluster_mask = cluster_labels == label
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 0:
            # For DBSCAN, points may not be contiguous, so take full range
            start_idx = cluster_indices.min()
            end_idx = cluster_indices.max() + 1
            
            segment_rssi = rssi[start_idx:end_idx]
            peak_idx_local = np.argmax(segment_rssi)
            duration = timestamps[end_idx-1] - timestamps[start_idx]
            
            # Calculate cluster metrics
            cluster_density = len(cluster_indices) / (end_idx - start_idx) if end_idx > start_idx else 1.0
            
            arcs.append({
                'timestamp': timestamps[start_idx:end_idx],
                'rssi': rssi[start_idx:end_idx],
                'phase': phase[start_idx:end_idx],
                'duration': duration,
                'peak_idx': peak_idx_local,
                'peak_rssi': segment_rssi[peak_idx_local],
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx-1],
                'cluster_id': int(label),
                'cluster_size': len(cluster_indices),
                'cluster_density': cluster_density,
                'noise_points': noise_points,
                'method': 'dbscan_advanced'
            })
    
    # Sort arcs by time
    arcs.sort(key=lambda x: x['start_time'])
    
    return arcs

def detect_arcs_within_pass(pass_data, method='peaks', min_prominence=0.05, min_width=5):
    """
    Detects individual arcs within a pass.
    
    :param pass_data: Dictionary with pass data
    :param method: Detection method ('peaks', 'derivative', 'threshold', 'dbscan', 'dbscan_advanced', 'auto')
    :param min_prominence: Minimum prominence for peak detection
    :param min_width: Minimum width for peak detection
    :return: List of detected arcs
    """
    timestamps = pass_data['timestamp']
    rssi = pass_data['rssi']
    phase = pass_data['phase']
    
    if len(timestamps) < 10:
        return [{
            'timestamp': timestamps,
            'rssi': rssi,
            'phase': phase,
            'duration': pass_data['duration'],
            'peak_idx': np.argmax(rssi),
            'peak_rssi': rssi[np.argmax(rssi)],
            'start_time': timestamps[0],
            'end_time': timestamps[-1]
        }]
    
    # Normalize RSSI for peak detection
    rssi_norm = (rssi - rssi.min()) / (rssi.max() - rssi.min()) if rssi.max() != rssi.min() else np.ones_like(rssi) * 0.5
    
    if method == 'peaks':
        return _detect_arcs_by_peaks_in_pass(timestamps, rssi, phase, rssi_norm, min_prominence, min_width)
    elif method == 'derivative':
        return _detect_arcs_by_derivative_in_pass(timestamps, rssi, phase, rssi_norm)
    elif method == 'threshold':
        return _detect_arcs_by_threshold_in_pass(timestamps, rssi, phase, rssi_norm)
    elif method == 'dbscan':
        return _detect_arcs_by_dbscan_in_pass(timestamps, rssi, phase, rssi_norm)
    elif method == 'dbscan_advanced':
        return _detect_arcs_by_dbscan_advanced_in_pass(timestamps, rssi, phase, rssi_norm)
    else:
        # Automatic method: try methods in preference order
        methods_to_try = ['peaks', 'dbscan_advanced', 'dbscan', 'derivative']
        for method_try in methods_to_try:
            try:
                if method_try == 'peaks':
                    arcs = _detect_arcs_by_peaks_in_pass(timestamps, rssi, phase, rssi_norm, min_prominence, min_width)
                elif method_try == 'dbscan':
                    arcs = _detect_arcs_by_dbscan_in_pass(timestamps, rssi, phase, rssi_norm)
                elif method_try == 'dbscan_advanced':
                    arcs = _detect_arcs_by_dbscan_advanced_in_pass(timestamps, rssi, phase, rssi_norm)
                elif method_try == 'derivative':
                    arcs = _detect_arcs_by_derivative_in_pass(timestamps, rssi, phase, rssi_norm)
                
                if len(arcs) > 0:
                    print(f"  Successful method: {method_try}")
                    return arcs
            except Exception as e:
                print(f"  Error with method {method_try}: {e}")
                continue
        
        # Final fallback
        return [{
            'timestamp': timestamps,
            'rssi': rssi,
            'phase': phase,
            'duration': pass_data['duration'],
            'peak_idx': np.argmax(rssi),
            'peak_rssi': rssi[np.argmax(rssi)],
            'start_time': timestamps[0],
            'end_time': timestamps[-1]
        }]

def detect_passes_and_arcs(interpolated_data, arc_detection_method='peaks', time_gap_threshold=2.0):
    """
    Main function that hierarchically detects passes and arcs.
    
    :param interpolated_data: Dictionary with interpolated data
    :param arc_detection_method: Method for arc detection
    :param time_gap_threshold: Threshold for pass separation
    :return: Tuple of (all_arcs, passes, pass_info)
    """
    # Step 1: Detect passes
    passes = detect_passes_from_interpolated(interpolated_data, time_gap_threshold)
    
    # Step 2: Detect arcs within each pass
    all_arcs = []
    pass_info = []
    
    for pass_idx, pass_data in enumerate(passes):
        print(f"\nProcessing pass {pass_idx + 1}/{len(passes)}...")
        
        # Detect arcs in this pass
        arcs_in_pass = detect_arcs_within_pass(pass_data, method=arc_detection_method)
        
        # Add metadata to each arc
        for arc_idx, arc in enumerate(arcs_in_pass):
            arc['pass_id'] = pass_idx
            arc['arc_in_pass'] = arc_idx
            arc['arc_global_id'] = len(all_arcs)
            all_arcs.append(arc)
        
        # Pass information
        pass_info.append({
            'pass_id': pass_idx,
            'start_time': pass_data['start_time'],
            'end_time': pass_data['end_time'],
            'duration': pass_data['duration'],
            'num_points': pass_data['num_points'],
            'num_arcs': len(arcs_in_pass),
            'method_used': arc_detection_method
        })
        
        print(f"  Pass {pass_idx + 1}: {len(arcs_in_pass)} arcs detected")
    
    print(f"\nDetection complete:")
    print(f"- {len(passes)} passes")
    print(f"- {len(all_arcs)} total arcs")
    
    return all_arcs, passes, pass_info

def plot_hierarchical_segmentation(interpolated_data, arcs, passes, pass_info):
    """
    Visualizes hierarchical segmentation: passes and arcs.
    
    :param interpolated_data: Original interpolated data
    :param arcs: List of detected arcs
    :param passes: List of detected passes
    :param pass_info: Information about passes
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    timestamps = interpolated_data['timestamp']
    rssi = interpolated_data['rssi']
    phase = interpolated_data['phase']
    
    # Plot 1: General view with passes
    axes[0].plot(timestamps, rssi, 'b-', alpha=0.3, label='Original RSSI')
    
    pass_colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for i, pass_data in enumerate(passes):
        color = pass_colors[i % len(pass_colors)]
        axes[0].plot(pass_data['timestamp'], pass_data['rssi'], 
                    color=color, linewidth=2, 
                    label=f'Pass {i+1} ({pass_data["duration"]:.1f}s)')
    
    axes[0].set_ylabel('RSSI (dBm)')
    axes[0].set_title(f'Detected Passes ({len(passes)} passes)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Arcs within passes
    axes[1].plot(timestamps, rssi, 'b-', alpha=0.3, label='Original RSSI')
    
    arc_colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    for i, arc in enumerate(arcs):
        color = arc_colors[arc['pass_id'] % len(arc_colors)]
        axes[1].plot(arc['timestamp'], arc['rssi'], 'o-',
                    color=color, linewidth=2, markersize=2,
                    label=f'P{arc["pass_id"]+1}-A{arc["arc_in_pass"]+1}')
        
        # Mark peak
        peak_idx = arc['peak_idx']
        axes[1].plot(arc['timestamp'][peak_idx], arc['rssi'][peak_idx], 
                    '*', color=color, markersize=10)
    
    axes[1].set_ylabel('RSSI (dBm)')
    axes[1].set_title(f'Detected Arcs ({len(arcs)} total arcs)')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Phase with segmentation
    axes[2].plot(timestamps, phase, 'g-', alpha=0.3, label='Original Phase')
    
    for i, arc in enumerate(arcs):
        color = arc_colors[arc['pass_id'] % len(arc_colors)]
        axes[2].plot(arc['timestamp'], arc['phase'], 'o-',
                    color=color, linewidth=2, markersize=2)
    
    axes[2].set_ylabel('Phase')
    axes[2].set_title('Phase with Segmentation')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    axes[3].axis('off')
    
    # Pass statistics
    pass_durations = [p['duration'] for p in pass_info]
    arcs_per_pass = [p['num_arcs'] for p in pass_info]
    
    stats_text = f"=== SEGMENTATION SUMMARY ===\n\n"
    stats_text += f"Detected passes: {len(passes)}\n"
    stats_text += f"Total arcs: {len(arcs)}\n"
    stats_text += f"Average duration per pass: {np.mean(pass_durations):.2f}s ± {np.std(pass_durations):.2f}s\n"
    stats_text += f"Average arcs per pass: {np.mean(arcs_per_pass):.1f} ± {np.std(arcs_per_pass):.1f}\n\n"
    
    stats_text += "=== DETAILS BY PASS ===\n"
    for i, info in enumerate(pass_info):
        stats_text += f"Pass {i+1}: {info['start_time']:.2f}s-{info['end_time']:.2f}s "
        stats_text += f"({info['duration']:.2f}s, {info['num_arcs']} arcs)\n"
    
    axes[3].text(0.05, 0.95, stats_text, transform=axes[3].transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output_plots', exist_ok=True)
    
    # Save figure
    plt.savefig(f'output_plots/hierarchical_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_dbscan_analysis(pass_data, method='dbscan_advanced'):
    """
    Visualizes DBSCAN analysis in detail.
    
    :param pass_data: Pass data to analyze
    :param method: DBSCAN method to use
    """
    timestamps = pass_data['timestamp']
    rssi = pass_data['rssi']
    phase = pass_data['phase']
    rssi_norm = (rssi - rssi.min()) / (rssi.max() - rssi.min()) if rssi.max() != rssi.min() else np.ones_like(rssi) * 0.5
    
    # Detect arcs with DBSCAN
    if method == 'dbscan_advanced':
        arcs = _detect_arcs_by_dbscan_advanced_in_pass(timestamps, rssi, phase, rssi_norm)
    else:
        arcs = _detect_arcs_by_dbscan_in_pass(timestamps, rssi, phase, rssi_norm)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: RSSI with clusters
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
    axes[0,0].plot(timestamps, rssi, 'k-', alpha=0.3, label='Original RSSI')
    
    for i, arc in enumerate(arcs):
        color = colors[i % len(colors)]
        axes[0,0].plot(arc['timestamp'], arc['rssi'], 'o-', 
                      color=color, linewidth=2, markersize=3,
                      label=f'Cluster {arc.get("cluster_id", i)}')
        
        # Mark peak
        peak_idx = arc['peak_idx']
        axes[0,0].plot(arc['timestamp'][peak_idx], arc['rssi'][peak_idx], 
                      '*', color=color, markersize=12)
    
    axes[0,0].set_ylabel('RSSI (dBm)')
    axes[0,0].set_title(f'DBSCAN - RSSI ({len(arcs)} clusters)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Phase with clusters
    axes[0,1].plot(timestamps, phase, 'k-', alpha=0.3, label='Original Phase')
    
    for i, arc in enumerate(arcs):
        color = colors[i % len(colors)]
        axes[0,1].plot(arc['timestamp'], arc['phase'], 'o-', 
                      color=color, linewidth=2, markersize=3)
    
    axes[0,1].set_ylabel('Phase')
    axes[0,1].set_title('DBSCAN - Phase')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Feature space (time vs RSSI)
    time_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    
    for i, arc in enumerate(arcs):
        color = colors[i % len(colors)]
        arc_time_norm = (arc['timestamp'] - timestamps.min()) / (timestamps.max() - timestamps.min())
        arc_rssi_norm = (arc['rssi'] - rssi.min()) / (rssi.max() - rssi.min()) if rssi.max() != rssi.min() else np.ones_like(arc['rssi']) * 0.5
        
        axes[1,0].scatter(arc_time_norm, arc_rssi_norm, 
                         color=color, s=30, alpha=0.7,
                         label=f'Cluster {arc.get("cluster_id", i)}')
    
    axes[1,0].set_xlabel('Normalized Time')
    axes[1,0].set_ylabel('Normalized RSSI')
    axes[1,0].set_title('Feature Space')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Cluster statistics
    axes[1,1].axis('off')
    
    stats_text = f"=== DBSCAN ANALYSIS ===\n\n"
    stats_text += f"Clusters found: {len(arcs)}\n"
    if len(arcs) > 0 and 'noise_points' in arcs[0]:
        stats_text += f"Noise points: {arcs[0]['noise_points']}\n"
    stats_text += f"Method used: {method}\n\n"
    
    stats_text += "=== DETAILS BY CLUSTER ===\n"
    for i, arc in enumerate(arcs):
        stats_text += f"Cluster {arc.get('cluster_id', i)}:\n"
        stats_text += f"  Duration: {arc['duration']:.2f}s\n"
        stats_text += f"  Peak RSSI: {arc['peak_rssi']:.2f} dBm\n"
        if 'cluster_size' in arc:
            stats_text += f"  Size: {arc['cluster_size']} points\n"
        if 'cluster_density' in arc:
            stats_text += f"  Density: {arc['cluster_density']:.3f}\n"
        stats_text += "\n"
    
    axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                  fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output_plots', exist_ok=True)
    
    # Save figure
    plt.savefig(f'output_plots/dbscan_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_arc_detection_methods_by_pass(passes):
    """
    Compares arc detection methods for each pass, including DBSCAN.
    
    :param passes: List of detected passes
    """
    methods = ['peaks', 'derivative', 'threshold', 'dbscan', 'dbscan_advanced']
    
    for pass_idx, pass_data in enumerate(passes):
        fig, axes = plt.subplots(len(methods), 1, figsize=(15, 12))
        
        if len(methods) == 1:
            axes = [axes]
        
        for i, method in enumerate(methods):
            try:
                arcs = detect_arcs_within_pass(pass_data, method=method)
                
                # Plot
                axes[i].plot(pass_data['timestamp'], pass_data['rssi'], 'b-', alpha=0.5, label='RSSI')
                
                colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
                for j, arc in enumerate(arcs):
                    color = colors[j % len(colors)]
                    axes[i].plot(arc['timestamp'], arc['rssi'], 'o-', 
                                color=color, linewidth=2, markersize=3)
                    
                    # Mark peak
                    peak_idx = arc['peak_idx']
                    axes[i].plot(arc['timestamp'][peak_idx], arc['rssi'][peak_idx], 
                                '*', color=color, markersize=10)
                
                # Additional information for DBSCAN
                method_info = method.capitalize()
                if len(arcs) > 0 and 'noise_points' in arcs[0]:
                    method_info += f" (noise: {arcs[0]['noise_points']})"
                
                axes[i].set_title(f'Pass {pass_idx+1} - {method_info} ({len(arcs)} arcs)')
                axes[i].set_ylabel('RSSI (dBm)')
                axes[i].grid(True, alpha=0.3)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error in {method}: {str(e)}', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'Pass {pass_idx+1} - {method.capitalize()} (ERROR)')
            
            if i == len(methods) - 1:
                axes[i].set_xlabel('Time (s)')
        
        plt.suptitle(f'Method Comparison (including DBSCAN) - Pass {pass_idx+1}')
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs('output_plots', exist_ok=True)
        
        # Save figure
        plt.savefig(f'output_plots/methods_comparison_pass_{pass_idx+1}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function demonstrating hierarchical segmentation capabilities.
    """
    print("=== HIERARCHICAL RFID SEGMENTATION DEMONSTRATION ===\n")
    
    # Load and process data
    print("Loading data from data/dynamic.csv...")
    tag_data = extract_tag_data('data/dynamic.csv')
    
    if not tag_data:
        print("Could not load data.")
        return
    
    # Select first tag
    tag_id = list(tag_data.keys())[0]
    tag_values = tag_data[tag_id]
    
    print(f"Processing tag: {tag_id}")
    print(f"Original samples: {len(tag_values['timestamp'])}")
    
    # Interpolate data
    print("Interpolating data...")
    interpolated = interpolate_tag_data(tag_values, num_points=100, kind='linear')
    
    # Hierarchical detection of passes and arcs
    print("Detecting passes and arcs...")
    arcs, passes, pass_info = detect_passes_and_arcs(interpolated, arc_detection_method='dbscan_advanced')
    
    # Visualize results
    print("\nGenerating visualizations...")
    plot_hierarchical_segmentation(interpolated, arcs, passes, pass_info)
    
    # Detailed DBSCAN analysis for first pass
    if len(passes) > 0:
        print("Detailed DBSCAN analysis for first pass...")
        plot_dbscan_analysis(passes[0], method='dbscan_advanced')
    
    # Compare methods for each pass (including DBSCAN)
    print("\nComparing arc detection methods (including DBSCAN)...")
    compare_arc_detection_methods_by_pass(passes)
    
    print(f"\nProcess completed:")
    print(f"- {len(passes)} passes detected")
    print(f"- {len(arcs)} total arcs detected")
    if len(passes) > 0:
        print(f"- Average of {len(arcs)/len(passes):.1f} arcs per pass")
    
    print(f"\nOutput files saved to 'output_plots/' directory:")
    print("- hierarchical_segmentation.png: Complete segmentation overview")
    print("- dbscan_analysis.png: Detailed DBSCAN cluster analysis")
    print("- methods_comparison_pass_*.png: Method comparison for each pass")

if __name__ == "__main__":
    main()