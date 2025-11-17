#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: timediff.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Advanced RFID tag data segmentation utilities using temporal analysis methods.
    Provides multiple algorithms for segmenting continuous RFID readings into
    individual passes/events including statistical thresholding, peak detection,
    machine learning clustering (GMM, DBSCAN), and adaptive methods with
    comprehensive visualization capabilities.

License: MIT License
Dependencies: numpy, scipy, sklearn, matplotlib, read_csv (local)
"""

import numpy as np
from typing import List, Tuple
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add src directory to path

from csv_data_loader import extract_tag_data

def split_data_by_time_diff(timestamps: List[float], threshold: float) -> List[List[float]]:
    """
    Splits a list of timestamps into segments based on time differences between readings.
    
    :param timestamps: Ordered list of timestamps
    :param threshold: Threshold to consider significant change in time between readings
    :return: List of lists, where each sublist contains timestamps from one segment
    """
    if not timestamps:
        return []
    
    segments = []
    current_segment = [timestamps[0]]
    avg_time_diff = np.mean(np.diff(timestamps))
    
    for i in range(1, len(timestamps)):
        time_diff = timestamps[i] - timestamps[i - 1]
        
        if time_diff > threshold * avg_time_diff:
            # If time difference is greater than threshold, start a new segment
            segments.append(current_segment)
            current_segment = [timestamps[i]]
        else:
            # Otherwise, add timestamp to current segment
            current_segment.append(timestamps[i])
                
    # Add the last segment
    if current_segment:
        segments.append(current_segment)
    return segments

def split_tag_data_by_time(tag_values, threshold=1.0):
    """
    Splits tag data into segments based on time jumps.
    
    :param tag_values: Dictionary with 'timestamp', 'rssi', 'phase' (numpy arrays)
    :param threshold: Multiplier threshold to detect time jump
    :return: List of dictionaries, each with data from one segment
    """
    timestamps = tag_values['timestamp']
    if len(timestamps) == 0:
        return []
    avg_time_diff = np.mean(np.diff(timestamps))
    deviation = np.std(np.diff(timestamps))
    split_indices = [0]
    for i in range(1, len(timestamps)):
        time_diff = timestamps[i] - timestamps[i - 1]
        if time_diff > avg_time_diff + threshold * deviation:
            split_indices.append(i)
    split_indices.append(len(timestamps))

    segments = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        segments.append({
            'timestamp': tag_values['timestamp'][start:end],
            'rssi': tag_values['rssi'][start:end],
            'phase': tag_values['phase'][start:end]
        })
    return segments

def split_tag_data_by_time_tukey(tag_values):
    """
    Splits tag data into segments using Tukey's criterion on time differences.
    
    :param tag_values: Dictionary with 'timestamp', 'rssi', 'phase' (numpy arrays)
    :return: List of dictionaries, each with data from one segment
    """
    timestamps = tag_values['timestamp']
    if len(timestamps) < 2:
        return [tag_values]

    diffs = np.diff(timestamps)
    q1 = np.percentile(diffs, 25)
    q3 = np.percentile(diffs, 75)
    iqr = q3 - q1
    tukey_threshold = q3 + 1.5 * iqr

    # Find indices where difference is an outlier
    split_indices = [0] + list(np.where(diffs > tukey_threshold)[0] + 1) + [len(timestamps)]

    segments = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        segments.append({
            'timestamp': tag_values['timestamp'][start:end],
            'rssi': tag_values['rssi'][start:end],
            'phase': tag_values['phase'][start:end]
        })
    return segments

def split_tag_data_by_rssi_minima(tag_values):
    """
    Splits tag data into segments using RSSI local minima as cut points.
    
    :param tag_values: Dictionary with 'timestamp', 'rssi', 'phase' (numpy arrays)
    :return: List of dictionaries, each with data from one segment
    """
    rssi = tag_values['rssi']
    # Find indices of local minima (excluding endpoints)
    minima_indices = np.where((rssi[1:-1] < rssi[:-2]) & (rssi[1:-1] < rssi[2:]))[0] + 1

    # Add start and end as possible cuts
    split_indices = [0] + minima_indices.tolist() + [len(rssi)]

    segments = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        segments.append({
            'timestamp': tag_values['timestamp'][start:end],
            'rssi': tag_values['rssi'][start:end],
            'phase': tag_values['phase'][start:end]
        })
    return segments

def split_tag_data_by_absolute_and_stat(tag_values, abs_threshold=1.0, stat_threshold=1.0):
    """
    First separates data by absolute time difference threshold (abs_threshold, in seconds).
    Then, in each segment, applies mean + stat_threshold * standard deviation method.
    
    :param tag_values: Dictionary with 'timestamp', 'rssi', 'phase' (numpy arrays)
    :param abs_threshold: Absolute time difference threshold (in seconds)
    :param stat_threshold: Standard deviation multiplier for second step
    :return: List of dictionaries, each with data from one subsegment
    """
    timestamps = tag_values['timestamp']
    rssi = tag_values['rssi']
    phase = tag_values['phase']

    # First cut: absolute value
    diffs = np.diff(timestamps)
    split_indices = [0] + list(np.where(diffs > abs_threshold)[0] + 1) + [len(timestamps)]

    all_segments = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        # Extract the segment
        seg_timestamps = timestamps[start:end]
        seg_rssi = rssi[start:end]
        seg_phase = phase[start:end]
        if len(seg_timestamps) < 2:
            continue

        # Second cut: mean + threshold * standard deviation
        seg_diffs = np.diff(seg_timestamps)
        avg_time_diff = np.mean(seg_diffs)
        deviation = np.std(seg_diffs)
        sub_split_indices = [0]
        for i in range(1, len(seg_timestamps)):
            time_diff = seg_timestamps[i] - seg_timestamps[i - 1]
            if time_diff > avg_time_diff + stat_threshold * deviation:
                sub_split_indices.append(i)
        sub_split_indices.append(len(seg_timestamps))

        for sub_start, sub_end in zip(sub_split_indices[:-1], sub_split_indices[1:]):
            if sub_end - sub_start < 2:
                continue
            
            # Calculate segment duration
            segment_timestamps = seg_timestamps[sub_start:sub_end]
            duration = segment_timestamps[-1] - segment_timestamps[0]
            
            all_segments.append({
                'timestamp': segment_timestamps,
                'rssi': seg_rssi[sub_start:sub_end],
                'phase': seg_phase[sub_start:sub_end],
                'duration': duration
            })
    return all_segments

import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

def split_tag_data_by_passes(tag_values, min_pass_duration=0.5, max_pass_duration=10.0, 
                           rssi_prominence=5.0, smoothing_window=5, min_samples=10):
    """
    Splits data into individual tag passes through the antenna.
    Detects RSSI peaks that represent complete passes.
    
    :param tag_values: Dictionary with 'timestamp', 'rssi', 'phase'
    :param min_pass_duration: Minimum pass duration (seconds)
    :param max_pass_duration: Maximum pass duration (seconds)
    :param rssi_prominence: Minimum RSSI peak prominence (dBm)
    :param smoothing_window: Window for signal smoothing
    :param min_samples: Minimum number of samples per segment
    :return: List of segments, each with one complete pass
    """
    timestamps = tag_values['timestamp']
    rssi = tag_values['rssi']
    phase = tag_values['phase']
    
    if len(timestamps) < min_samples:
        return []
    
    # 1. Smooth RSSI signal to reduce noise
    if len(rssi) >= smoothing_window:
        rssi_smooth = savgol_filter(rssi, min(smoothing_window, len(rssi)//2*2-1), 2)
    else:
        rssi_smooth = rssi
    
    # 2. Find RSSI peaks (local maxima)
    peaks, properties = find_peaks(rssi_smooth, 
                                 prominence=rssi_prominence,
                                 width=min_samples//2,
                                 distance=min_samples)
    
    if len(peaks) == 0:
        # If no clear peaks, use temporal method as fallback
        return split_tag_data_by_time(tag_values, threshold=2.0)
    
    segments = []
    
    for peak_idx in peaks:
        peak_time = timestamps[peak_idx]
        peak_rssi = rssi[peak_idx]
        
        # 3. Find pass boundaries around the peak
        # Search backwards from peak until finding valley or significant change
        start_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            # Criteria for pass start:
            # - RSSI drops significantly
            # - Large time gap
            # - Reached beginning
            if (rssi_smooth[peak_idx] - rssi_smooth[i] > rssi_prominence/2 or
                timestamps[peak_idx] - timestamps[i] > max_pass_duration or
                i == 0):
                start_idx = i
                break
            start_idx = i
        
        # Search forwards from peak until finding valley
        end_idx = peak_idx
        for i in range(peak_idx + 1, len(rssi_smooth)):
            # Criteria for pass end:
            # - RSSI drops significantly
            # - Large time gap
            # - Reached end
            if (rssi_smooth[peak_idx] - rssi_smooth[i] > rssi_prominence/2 or
                timestamps[i] - timestamps[peak_idx] > max_pass_duration or
                i == len(rssi_smooth) - 1):
                end_idx = i + 1
                break
            end_idx = i + 1
        
        # 4. Verify duration is within limits
        duration = timestamps[end_idx-1] - timestamps[start_idx]
        if (duration >= min_pass_duration and 
            duration <= max_pass_duration and
            end_idx - start_idx >= min_samples):
            
            segments.append({
                'timestamp': timestamps[start_idx:end_idx],
                'rssi': rssi[start_idx:end_idx],
                'phase': phase[start_idx:end_idx],
                'peak_idx': peak_idx - start_idx,  # Peak index within segment
                'peak_rssi': peak_rssi,
                'duration': duration
            })
    
    return segments

def split_tag_data_by_valleys(tag_values, min_pass_duration=0.5, max_gap=2.0):
    """
    Alternative method: splits by RSSI valleys (minima).
    Useful when passes are well separated.
    
    :param tag_values: Dictionary with 'timestamp', 'rssi', 'phase'
    :param min_pass_duration: Minimum pass duration (seconds)
    :param max_gap: Maximum time gap (seconds)
    :return: List of segments with individual passes
    """
    timestamps = tag_values['timestamp']
    rssi = tag_values['rssi']
    phase = tag_values['phase']
    
    if len(timestamps) < 5:
        return []
    
    # Smooth the signal
    if len(rssi) >= 5:
        rssi_smooth = savgol_filter(rssi, min(5, len(rssi)//2*2-1), 2)
    else:
        rssi_smooth = rssi
    
    # Find local minima (valleys)
    valleys, _ = find_peaks(-rssi_smooth, distance=10)
    
    # Add start and end as cut points
    split_points = [0] + valleys.tolist() + [len(timestamps)]
    split_points = sorted(set(split_points))  # Remove duplicates and sort
    
    segments = []
    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]
        
        if end_idx - start_idx < 5:  # Too few points
            continue
            
        duration = timestamps[end_idx-1] - timestamps[start_idx]
        if duration >= min_pass_duration:
            segments.append({
                'timestamp': timestamps[start_idx:end_idx],
                'rssi': rssi[start_idx:end_idx],
                'phase': phase[start_idx:end_idx],
                'duration': duration
            })
    
    return segments

def visualize_segmentation(tag_values, segments, title="Pass Segmentation"):
    """
    Visualizes detected segments over the original signal.
    
    :param tag_values: Original tag data
    :param segments: List of detected segments
    :param title: Plot title
    """
    plt.figure(figsize=(15, 8))
    
    # Original signal
    plt.subplot(2, 1, 1)
    plt.plot(tag_values['timestamp'], tag_values['rssi'], 'b-', alpha=0.7, label='Original RSSI')
    
    # Mark the segments
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        plt.plot(segment['timestamp'], segment['rssi'], 'o-', 
                color=color, label=f'Pass {i+1} ({segment["duration"]:.1f}s)', 
                linewidth=2, markersize=4)
        
        # Mark peak if it exists
        if 'peak_idx' in segment:
            peak_idx = segment['peak_idx']
            plt.plot(segment['timestamp'][peak_idx], segment['rssi'][peak_idx], 
                    '*', color=color, markersize=12)
    
    plt.xlabel('Time (s)')
    plt.ylabel('RSSI (dBm)')
    plt.title(f'{title} - {len(segments)} passes detected')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Duration histogram
    plt.subplot(2, 1, 2)
    durations = [seg['duration'] for seg in segments]
    plt.hist(durations, bins=min(10, len(durations)), alpha=0.7, edgecolor='black')
    plt.xlabel('Pass Duration (s)')
    plt.ylabel('Frequency')
    plt.title('Pass Duration Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print(f"Passes detected: {len(segments)}")
    if durations:
        print(f"Average duration: {np.mean(durations):.2f}s ± {np.std(durations):.2f}s")
        print(f"Duration min/max: {np.min(durations):.2f}s / {np.max(durations):.2f}s")

def smart_segmentation(tag_values, method='peaks'):
    """
    Smart segmentation that chooses the best method according to data.
    
    :param tag_values: Tag data dictionary
    :param method: Segmentation method ('peaks', 'valleys', 'auto')
    :return: List of detected segments
    """
    if method == 'peaks':
        segments = split_tag_data_by_passes(tag_values)
    elif method == 'valleys':
        segments = split_tag_data_by_valleys(tag_values)
    else:  # auto
        # Try both methods and choose the one with better results
        segments_peaks = split_tag_data_by_passes(tag_values)
        segments_valleys = split_tag_data_by_valleys(tag_values)
        
        # Criterion: the one with more uniform pass durations
        if len(segments_peaks) > 0 and len(segments_valleys) > 0:
            std_peaks = np.std([s['duration'] for s in segments_peaks])
            std_valleys = np.std([s['duration'] for s in segments_valleys])
            segments = segments_peaks if std_peaks < std_valleys else segments_valleys
        elif len(segments_peaks) > 0:
            segments = segments_peaks
        else:
            segments = segments_valleys
    
    return segments

from sklearn.mixture import GaussianMixture
import numpy as np

def split_tag_data_by_gmm(tag_values, min_segment_size=5):
    """
    Uses Gaussian Mixture Model to detect two distributions in time differences:
    - Distribution 1: Small differences (within pass)
    - Distribution 2: Large differences (between passes)
    
    :param tag_values: Tag data dictionary
    :param min_segment_size: Minimum segment size
    :return: List of detected segments
    """
    timestamps = tag_values['timestamp']
    rssi = tag_values['rssi']
    phase = tag_values['phase']
    
    if len(timestamps) < min_segment_size:
        return [tag_values]
    
    # Calculate time differences
    diffs = np.diff(timestamps)
    
    if len(diffs) < 3:
        return [tag_values]
    
    # Apply GMM with 2 components
    diffs_reshaped = diffs.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(diffs_reshaped)
    
    # Predict which component each difference belongs to
    labels = gmm.predict(diffs_reshaped)
    
    # Identify which component corresponds to large gaps
    means = gmm.means_.flatten()
    large_gap_component = np.argmax(means)  # Component with largest mean
    
    # Find indices where there are large gaps
    gap_indices = np.where(labels == large_gap_component)[0] + 1  # +1 because diffs has one less element
    
    # Create cut points
    split_indices = [0] + gap_indices.tolist() + [len(timestamps)]
    split_indices = sorted(set(split_indices))
    
    # Create segments
    segments = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        if end - start >= min_segment_size:
            segment_timestamps = timestamps[start:end]
            duration = segment_timestamps[-1] - segment_timestamps[0]
            
            segments.append({
                'timestamp': segment_timestamps,
                'rssi': rssi[start:end],
                'phase': phase[start:end],
                'duration': duration
            })
    
    return segments

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def split_tag_data_by_dbscan(tag_values, eps=0.5, min_samples=3):
    """
    Uses DBSCAN to cluster similar time differences.
    
    :param tag_values: Tag data dictionary
    :param eps: DBSCAN epsilon parameter
    :param min_samples: DBSCAN minimum samples parameter
    :return: List of detected segments
    """
    timestamps = tag_values['timestamp']
    rssi = tag_values['rssi']
    phase = tag_values['phase']
    
    if len(timestamps) < 5:
        return [tag_values]
    
    # Calculate time differences
    diffs = np.diff(timestamps)
    
    # Normalize differences for DBSCAN
    scaler = StandardScaler()
    diffs_scaled = scaler.fit_transform(diffs.reshape(-1, 1))
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(diffs_scaled)
    
    # Find cluster with largest differences (outliers or separate cluster)
    unique_labels = set(labels)
    if -1 in unique_labels:  # -1 are outliers in DBSCAN
        outlier_indices = np.where(labels == -1)[0] + 1
    else:
        # If no outliers, use cluster with largest mean
        cluster_means = []
        for label in unique_labels:
            if label != -1:
                cluster_diffs = diffs[labels == label]
                cluster_means.append(np.mean(cluster_diffs))
        
        max_cluster = unique_labels[np.argmax(cluster_means)] if cluster_means else 0
        outlier_indices = np.where(labels == max_cluster)[0] + 1
    
    # Create segments
    split_indices = [0] + outlier_indices.tolist() + [len(timestamps)]
    split_indices = sorted(set(split_indices))
    
    segments = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        if end - start >= 2:
            segment_timestamps = timestamps[start:end]
            duration = segment_timestamps[-1] - segment_timestamps[0]
            
            segments.append({
                'timestamp': segment_timestamps,
                'rssi': rssi[start:end],
                'phase': phase[start:end],
                'duration': duration
            })
    
    return segments

def jenks_breaks(data, n_classes=2):
    """
    Simple implementation of Jenks Natural Breaks to find optimal threshold.
    
    :param data: Data array to classify
    :param n_classes: Number of classes
    :return: List of break points
    """
    if len(data) <= n_classes:
        return [min(data), max(data)]
    
    data = np.sort(data)
    n = len(data)
    
    # Variance matrix
    variance_combinations = np.zeros((n_classes, n))
    
    # Fill first row (single class)
    for i in range(n):
        variance_combinations[0, i] = np.var(data[:i+1]) * (i + 1)
    
    # Fill remaining rows
    for i in range(1, n_classes):
        for j in range(i, n):
            min_var = float('inf')
            for k in range(i-1, j):
                var = variance_combinations[i-1, k] + np.var(data[k+1:j+1]) * (j - k)
                if var < min_var:
                    min_var = var
            variance_combinations[i, j] = min_var
    
    # Find break points
    breaks = []
    j = n - 1
    for i in range(n_classes - 1, 0, -1):
        for k in range(i-1, j):
            if variance_combinations[i, j] == variance_combinations[i-1, k] + np.var(data[k+1:j+1]) * (j - k):
                breaks.append(data[k])
                j = k
                break
    
    breaks.append(data[0])
    return sorted(breaks)

def split_tag_data_by_jenks(tag_values, min_segment_size=3):
    """
    Uses Jenks Natural Breaks to find optimal threshold between small and large differences.
    
    :param tag_values: Tag data dictionary
    :param min_segment_size: Minimum segment size
    :return: List of detected segments
    """
    timestamps = tag_values['timestamp']
    rssi = tag_values['rssi']
    phase = tag_values['phase']
    
    if len(timestamps) < min_segment_size:
        return [tag_values]
    
    diffs = np.diff(timestamps)
    
    if len(diffs) < 3:
        return [tag_values]
    
    # Find optimal cut point
    breaks = jenks_breaks(diffs, n_classes=2)
    threshold = breaks[1]  # Threshold between two classes
    
    # Apply threshold
    split_indices = [0] + list(np.where(diffs > threshold)[0] + 1) + [len(timestamps)]
    split_indices = sorted(set(split_indices))
    
    segments = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        if end - start >= min_segment_size:
            segment_timestamps = timestamps[start:end]
            duration = segment_timestamps[-1] - segment_timestamps[0]
            
            segments.append({
                'timestamp': segment_timestamps,
                'rssi': rssi[start:end],
                'phase': phase[start:end],
                'duration': duration
            })
    
    return segments

def split_tag_data_by_adaptive_tukey(tag_values, percentile_low=10, percentile_high=90):
    """
    Improved version of Tukey method using adaptive percentiles.
    
    :param tag_values: Tag data dictionary
    :param percentile_low: Lower percentile for threshold calculation
    :param percentile_high: Upper percentile for threshold calculation
    :return: List of detected segments
    """
    timestamps = tag_values['timestamp']
    rssi = tag_values['rssi']
    phase = tag_values['phase']
    
    if len(timestamps) < 3:
        return [tag_values]
    
    diffs = np.diff(timestamps)
    
    # Use more extreme percentiles to better capture bimodality
    q_low = np.percentile(diffs, percentile_low)
    q_high = np.percentile(diffs, percentile_high)
    iqr = q_high - q_low
    
    # More aggressive threshold for gap detection
    threshold = q_high + 0.5 * iqr  # More sensitive than traditional 1.5
    
    split_indices = [0] + list(np.where(diffs > threshold)[0] + 1) + [len(timestamps)]
    split_indices = sorted(set(split_indices))
    
    segments = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        if end - start >= 2:
            segment_timestamps = timestamps[start:end]
            duration = segment_timestamps[-1] - segment_timestamps[0]
            
            segments.append({
                'timestamp': segment_timestamps,
                'rssi': rssi[start:end],
                'phase': phase[start:end],
                'duration': duration
            })
    
    return segments

def smart_split_improved(tag_values, method='auto'):
    """
    Function that tries different methods and chooses the best result.
    
    :param tag_values: Tag data dictionary
    :param method: Segmentation method or 'auto' for automatic selection
    :return: List of detected segments
    """
    methods = {
        'gmm': split_tag_data_by_gmm,
        'dbscan': split_tag_data_by_dbscan,
        'jenks': split_tag_data_by_jenks,
        'adaptive_tukey': split_tag_data_by_adaptive_tukey,
        'original': lambda x: split_tag_data_by_absolute_and_stat(x, 1.0, 1.0)
    }
    
    if method != 'auto':
        return methods[method](tag_values)
    
    # Try all methods
    results = {}
    for name, func in methods.items():
        try:
            segments = func(tag_values)
            if segments:
                durations = [s['duration'] for s in segments]
                # Quality metric: duration uniformity + reasonable number of segments
                quality = 1 / (np.std(durations) + 1e-6) * np.log(len(segments) + 1)
                results[name] = (segments, quality)
        except:
            continue
    
    # Choose best method
    if results:
        best_method = max(results.keys(), key=lambda k: results[k][1])
        return results[best_method][0]
    else:
        return [tag_values]

if __name__ == "__main__":
        
    # Example usage with tag data
    tags_data = extract_tag_data('data/dynamic.csv')
    tag_id = list(tags_data.keys())[0]
    tag_values = tags_data[tag_id]
    
    # Time differences histogram
    import matplotlib.pyplot as plt
    
    diffs = np.diff(tag_values['timestamp'])
    plt.hist(diffs, bins=200, alpha=0.7, color='blue')
    plt.xlabel('Time Difference (s)')
    plt.ylabel('Frequency')
    plt.title(f'Time Differences Histogram for Tag {tag_id}')
    plt.grid()
    plt.show()

    # Split tag data by time
    segments = split_tag_data_by_gmm(tag_values)
    visualize_segmentation(tag_values, segments)
    print(f"Segments found: {len(segments)}")

    # Plot segments and complete dataset
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.scatter(tag_values['timestamp'], tag_values['rssi'], label='RSSI', color='blue')
    plt.scatter(tag_values['timestamp'], tag_values['phase'], label='Phase', color='orange')
    plt.xlabel('Timestamp')
    plt.ylabel('Values')
    plt.title(f'Tag ID: {tag_id} - Complete Data')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot each segment separately
    import matplotlib.pyplot as plt

    # Alternating color palette
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    plt.figure(figsize=(12, 6))
    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        plt.scatter(segment['timestamp'], segment['rssi'], color=color, label=f'Segment {i+1}', s=20, alpha=0.7)

    plt.xlabel('Timestamp')
    plt.ylabel('RSSI')
    plt.title(f'Tag ID: {tag_id} - RSSI Segments (Timeline)')
    # plt.legend()
    plt.grid()
    plt.show()

