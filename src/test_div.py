#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: test_div.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Advanced RFID tag segmentation and arc detection system. Processes CSV files
    containing RFID data to detect individual passes and segment them into arcs
    using interpolation, smoothing, and local minima detection. Provides comprehensive
    visualization and statistical analysis of detected patterns.

License: MIT License
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, argrelmin
from timediff import split_tag_data_by_absolute_and_stat

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add src directory to path

from csv_data_loader import extract_tag_data

def interpolate_and_smooth_segment(segment, num_points=200, smoothing_sigma=1.0):
    """
    Interpolates and smooths an individual segment.
    
    :param segment: Dictionary with 'timestamp', 'rssi', 'phase'
    :param num_points: Number of points for interpolation
    :param smoothing_sigma: Gaussian smoothing parameter
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
            'original_segment': segment
        }
    
    # Create uniform temporal grid
    t_min = timestamps.min()
    t_max = timestamps.max()
    t_uniform = np.linspace(t_min, t_max, num_points)
    
    try:
        # Interpolate RSSI
        interp_func_rssi = interp1d(timestamps, rssi, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
        rssi_interpolated = interp_func_rssi(t_uniform)
        
        # Interpolate Phase
        interp_func_phase = interp1d(timestamps, phase, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        phase_interpolated = interp_func_phase(t_uniform)
        
        # Apply smoothing
        rssi_smooth = gaussian_filter1d(rssi_interpolated, sigma=smoothing_sigma)
        phase_smooth = gaussian_filter1d(phase_interpolated, sigma=smoothing_sigma)
        
        return {
            'timestamp': t_uniform,
            'rssi': rssi_interpolated,
            'phase': phase_interpolated,
            'rssi_smooth': rssi_smooth,
            'phase_smooth': phase_smooth,
            'original_segment': segment
        }
        
    except Exception as e:
        print(f"Error in interpolation: {e}")
        return {
            'timestamp': timestamps,
            'rssi': rssi,
            'phase': phase,
            'rssi_smooth': rssi,
            'phase_smooth': phase,
            'original_segment': segment
        }

def detect_local_minima(signal, min_distance=10, prominence_threshold=0.1):
    """
    Detects local minima in a smoothed signal.
    
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

def segment_pass_into_arcs(interpolated_segment, min_arc_duration=0.1):
    """
    Segments an interpolated pass into individual arcs based on local minima.
    
    :param interpolated_segment: Result of interpolate_and_smooth_segment()
    :param min_arc_duration: Minimum arc duration (seconds)
    :return: List of arcs
    """
    timestamps = interpolated_segment['timestamp']
    rssi = interpolated_segment['rssi']
    phase = interpolated_segment['phase']
    rssi_smooth = interpolated_segment['rssi_smooth']
    phase_smooth = interpolated_segment['phase_smooth']
    
    # Detect local minima
    minima_indices = detect_local_minima(rssi_smooth)
    
    # If no clear local minima, check if it's already a single arc
    if len(minima_indices) == 0:
        # Check if extremes are minima (signal forms complete arc)
        start_is_min = rssi_smooth[0] < rssi_smooth[len(rssi_smooth)//4]
        end_is_min = rssi_smooth[-1] < rssi_smooth[3*len(rssi_smooth)//4]
        
        if start_is_min or end_is_min:
            # Already a complete arc, don't divide
            return [{
                'timestamp': timestamps,
                'rssi': rssi,
                'phase': phase,
                'rssi_smooth': rssi_smooth,
                'phase_smooth': phase_smooth,
                'duration': timestamps[-1] - timestamps[0],
                'peak_idx': np.argmax(rssi_smooth),
                'peak_rssi': rssi_smooth[np.argmax(rssi_smooth)],
                'is_complete_arc': True,
                'start_time': timestamps[0],
                'end_time': timestamps[-1]
            }]
        else:
            # Search for less restrictive minima
            minima_indices = detect_local_minima(rssi_smooth, min_distance=5, prominence_threshold=0.05)
    
    # If still no minima, return entire pass as one arc
    if len(minima_indices) == 0:
        return [{
            'timestamp': timestamps,
            'rssi': rssi,
            'phase': phase,
            'rssi_smooth': rssi_smooth,
            'phase_smooth': phase_smooth,
            'duration': timestamps[-1] - timestamps[0],
            'peak_idx': np.argmax(rssi_smooth),
            'peak_rssi': rssi_smooth[np.argmax(rssi_smooth)],
            'is_complete_arc': False,
            'start_time': timestamps[0],
            'end_time': timestamps[-1]
        }]
    
    # Add start and end as split points if not included
    split_indices = [0] + minima_indices.tolist() + [len(timestamps)]
    split_indices = sorted(set(split_indices))  # Remove duplicates and sort
    
    arcs = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        
        if end_idx - start_idx < 5:  # Too few points
            continue
            
        arc_timestamps = timestamps[start_idx:end_idx]
        arc_duration = arc_timestamps[-1] - arc_timestamps[0]
        
        if arc_duration >= min_arc_duration:
            arc_rssi = rssi[start_idx:end_idx]
            arc_phase = phase[start_idx:end_idx]
            arc_rssi_smooth = rssi_smooth[start_idx:end_idx]
            arc_phase_smooth = phase_smooth[start_idx:end_idx]
            
            # Find peak within the arc
            peak_idx_local = np.argmax(arc_rssi_smooth)
            
            # Check if it's a complete arc (minima at extremes)
            is_complete = (start_idx in minima_indices or start_idx == 0) and \
                         (end_idx-1 in minima_indices or end_idx == len(timestamps))
            
            arcs.append({
                'timestamp': arc_timestamps,
                'rssi': arc_rssi,
                'phase': arc_phase,
                'rssi_smooth': arc_rssi_smooth,
                'phase_smooth': arc_phase_smooth,
                'duration': arc_duration,
                'peak_idx': peak_idx_local,
                'peak_rssi': arc_rssi_smooth[peak_idx_local],
                'is_complete_arc': is_complete,
                'start_time': arc_timestamps[0],
                'end_time': arc_timestamps[-1],
                'min_start': start_idx in minima_indices,
                'min_end': (end_idx-1) in minima_indices
            })
    
    return arcs

def process_segmentation_to_arcs(all_segmentations, num_interp_points=200, smoothing_sigma=1.0):
    """
    Processes all segmentations to convert passes into individual arcs.
    
    :param all_segmentations: Result of load_and_segment_csv_files()
    :param num_interp_points: Number of points for interpolation
    :param smoothing_sigma: Smoothing parameter
    :return: Dictionary with arcs by file and tag
    """
    all_arcs = {}
    
    for csv_file, file_data in all_segmentations.items():
        print(f"\nProcessing arcs for: {os.path.basename(csv_file)}")
        
        file_arcs = {}
        
        for tag_id, tag_data in file_data.items():
            segments = tag_data['segments']
            print(f"  Tag {tag_id}: {len(segments)} passes -> ", end="")
            
            tag_arcs = []
            
            for segment_idx, segment in enumerate(segments):
                # Interpolate and smooth the pass
                interpolated = interpolate_and_smooth_segment(
                    segment, 
                    num_points=num_interp_points, 
                    smoothing_sigma=smoothing_sigma
                )
                
                # Segment into arcs
                arcs_in_segment = segment_pass_into_arcs(interpolated)
                
                # Add context information
                for arc_idx, arc in enumerate(arcs_in_segment):
                    arc['pass_id'] = segment_idx
                    arc['arc_in_pass'] = arc_idx
                    arc['total_arcs_in_pass'] = len(arcs_in_segment)
                
                tag_arcs.extend(arcs_in_segment)
            
            file_arcs[tag_id] = {
                'original_data': tag_data['original_data'],
                'original_segments': segments,
                'arcs': tag_arcs,
                'num_arcs': len(tag_arcs)
            }
            
            print(f"{len(tag_arcs)} arcs")
        
        all_arcs[csv_file] = file_arcs
    
    return all_arcs

def load_and_segment_csv_files(data_folder='data', abs_threshold=1.0, stat_threshold=1.0):
    """
    Loads multiple CSV files and segments data based on absolute time differences.
    
    :param data_folder: Folder containing CSV files
    :param abs_threshold: Absolute time difference threshold (in seconds)
    :param stat_threshold: Standard deviation multiplier
    :return: Dictionary with all segments by file and tag
    """
    all_segmentations = {}
    
    # Search for all CSV files in folder
    csv_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        
        try:
            # Extract data from CSV
            tag_data = extract_tag_data(csv_file)
            
            if not tag_data:
                print(f"  Could not extract data from {csv_file}")
                continue
            
            file_segments = {}
            
            # Segment each tag in the file
            for tag_id, tag_values in tag_data.items():
                print(f"  Tag {tag_id}: {len(tag_values['timestamp'])} samples")
                
                # Apply segmentation
                segments = split_tag_data_by_absolute_and_stat(
                    tag_values, 
                    abs_threshold=abs_threshold, 
                    stat_threshold=stat_threshold
                )
                
                file_segments[tag_id] = {
                    'original_data': tag_values,
                    'segments': segments,
                    'num_segments': len(segments)
                }
                
                print(f"    -> {len(segments)} segments found")
            
            all_segmentations[csv_file] = file_segments
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
    
    return all_segmentations

def plot_segmentation_for_file(csv_file, file_data, output_dir='output_plots'):
    """
    Creates a plot with all segments for all tags in a file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of tags in the file
    num_tags = len(file_data)
    
    if num_tags == 0:
        return
    
    # Create figure with subplots for each tag
    fig, axes = plt.subplots(num_tags, 2, figsize=(20, 6*num_tags))
    
    # If only one tag, axes won't be a 2D array
    if num_tags == 1:
        axes = [axes]
    
    # Colors for segments
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for tag_idx, (tag_id, tag_data) in enumerate(file_data.items()):
        original_data = tag_data['original_data']
        segments = tag_data['segments']
        
        # Plot RSSI
        ax_rssi = axes[tag_idx][0]
        ax_rssi.plot(original_data['timestamp'], original_data['rssi'], 
                    'k-', alpha=0.3, linewidth=0.5, label='Original RSSI')
        
        # Color each segment
        for i, segment in enumerate(segments):
            color = colors[i % len(colors)]
            ax_rssi.plot(segment['timestamp'], segment['rssi'], 
                        color=color, linewidth=2, alpha=0.8, 
                        label=f'Segment {i+1} ({len(segment["timestamp"])} pts)')
        
        ax_rssi.set_xlabel('Time (s)')
        ax_rssi.set_ylabel('RSSI (dBm)')
        ax_rssi.set_title(f'Tag {tag_id} - Segmented RSSI ({len(segments)} segments)')
        ax_rssi.grid(True, alpha=0.3)
        ax_rssi.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot Phase
        ax_phase = axes[tag_idx][1]
        ax_phase.plot(original_data['timestamp'], original_data['phase'], 
                     'k-', alpha=0.3, linewidth=0.5, label='Original Phase')
        
        # Color each segment
        for i, segment in enumerate(segments):
            color = colors[i % len(colors)]
            ax_phase.plot(segment['timestamp'], segment['phase'], 
                         color=color, linewidth=2, alpha=0.8, 
                         label=f'Segment {i+1}')
        
        ax_phase.set_xlabel('Time (s)')
        ax_phase.set_ylabel('Phase')
        ax_phase.set_title(f'Tag {tag_id} - Segmented Phase')
        ax_phase.grid(True, alpha=0.3)
        ax_phase.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # General title
    file_name = os.path.basename(csv_file)
    fig.suptitle(f'Segmentation of {file_name}', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save figure
    safe_filename = file_name.replace('.csv', '').replace('/', '_').replace('\\', '_')
    output_path = os.path.join(output_dir, f'segmentation_{safe_filename}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {output_path}")
    
    plt.show()

def plot_summary_statistics(all_segmentations, output_dir='output_plots'):
    """
    Creates a summary plot with segmentation statistics.
    """
    # Collect statistics
    file_names = []
    num_segments_per_file = []
    durations_all = []
    num_samples_all = []
    
    for csv_file, file_data in all_segmentations.items():
        file_name = os.path.basename(csv_file)
        file_names.append(file_name)
        
        total_segments = 0
        for tag_id, tag_data in file_data.items():
            segments = tag_data['segments']
            total_segments += len(segments)
            
            # Collect durations and number of samples
            for segment in segments:
                duration = segment['timestamp'][-1] - segment['timestamp'][0]
                durations_all.append(duration)
                num_samples_all.append(len(segment['timestamp']))
        
        num_segments_per_file.append(total_segments)
    
    # Create figure with statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Number of segments per file
    axes[0,0].bar(range(len(file_names)), num_segments_per_file)
    axes[0,0].set_xlabel('Files')
    axes[0,0].set_ylabel('Total number of segments')
    axes[0,0].set_title('Segments per file')
    axes[0,0].set_xticks(range(len(file_names)))
    axes[0,0].set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                              for name in file_names], rotation=45)
    
    # Duration histogram
    axes[0,1].hist(durations_all, bins=20, alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Segment duration (s)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title(f'Duration distribution\n(Mean: {np.mean(durations_all):.2f}s)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Number of samples histogram
    axes[1,0].hist(num_samples_all, bins=20, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Number of samples per segment')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title(f'Size distribution\n(Mean: {np.mean(num_samples_all):.1f} samples)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Textual statistics
    axes[1,1].axis('off')
    stats_text = f"""=== GLOBAL SUMMARY ===

Files processed: {len(all_segmentations)}
Total segments: {len(durations_all)}

Segment duration:
  Mean: {np.mean(durations_all):.3f} s
  Median: {np.median(durations_all):.3f} s
  Std dev: {np.std(durations_all):.3f} s
  Min: {np.min(durations_all):.3f} s
  Max: {np.max(durations_all):.3f} s

Samples per segment:
  Mean: {np.mean(num_samples_all):.1f}
  Median: {np.median(num_samples_all):.1f}
  Std dev: {np.std(num_samples_all):.1f}
  Min: {np.min(num_samples_all)}
  Max: {np.max(num_samples_all)}"""
    
    axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save summary
    summary_path = os.path.join(output_dir, 'segmentation_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Summary saved: {summary_path}")
    
    plt.show()

def plot_arcs_for_file(csv_file, file_data, output_dir='output_plots'):
    """
    Creates a plot showing arc segmentation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_tags = len(file_data)
    if num_tags == 0:
        return
    
    fig, axes = plt.subplots(num_tags, 2, figsize=(20, 6*num_tags))
    if num_tags == 1:
        axes = [axes]
    
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for tag_idx, (tag_id, tag_data) in enumerate(file_data.items()):
        original_data = tag_data['original_data']
        arcs = tag_data['arcs']
        
        # Plot RSSI with arcs
        ax_rssi = axes[tag_idx][0]
        ax_rssi.plot(original_data['timestamp'], original_data['rssi'], 
                    'k-', alpha=0.2, linewidth=0.5, label='Original RSSI')
        
        # Color each arc
        for i, arc in enumerate(arcs):
            color = colors[i % len(colors)]
            
            # Smoothed data
            ax_rssi.plot(arc['timestamp'], arc['rssi_smooth'], 
                        color=color, linewidth=2, alpha=0.8, 
                        label=f'Arc {i+1} (P{arc["pass_id"]+1})')
            
            # Mark the peak
            peak_idx = arc['peak_idx']
            ax_rssi.plot(arc['timestamp'][peak_idx], arc['rssi_smooth'][peak_idx], 
                        '*', color=color, markersize=12)
            
            # Mark if complete arc
            if arc['is_complete_arc']:
                ax_rssi.plot(arc['timestamp'][0], arc['rssi_smooth'][0], 
                            'v', color=color, markersize=8, alpha=0.7)
                ax_rssi.plot(arc['timestamp'][-1], arc['rssi_smooth'][-1], 
                            'v', color=color, markersize=8, alpha=0.7)
        
        ax_rssi.set_xlabel('Time (s)')
        ax_rssi.set_ylabel('RSSI (dBm)')
        ax_rssi.set_title(f'Tag {tag_id} - Detected Arcs ({len(arcs)} arcs)')
        ax_rssi.grid(True, alpha=0.3)
        ax_rssi.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot Phase with arcs
        ax_phase = axes[tag_idx][1]
        ax_phase.plot(original_data['timestamp'], original_data['phase'], 
                     'k-', alpha=0.2, linewidth=0.5, label='Original Phase')
        
        for i, arc in enumerate(arcs):
            color = colors[i % len(colors)]
            ax_phase.plot(arc['timestamp'], arc['phase_smooth'], 
                         color=color, linewidth=2, alpha=0.8)
        
        ax_phase.set_xlabel('Time (s)')
        ax_phase.set_ylabel('Phase')
        ax_phase.set_title(f'Tag {tag_id} - Smoothed Phase')
        ax_phase.grid(True, alpha=0.3)
    
    file_name = os.path.basename(csv_file)
    fig.suptitle(f'Arc Segmentation - {file_name}', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save figure
    safe_filename = file_name.replace('.csv', '').replace('/', '_').replace('\\', '_')
    output_path = os.path.join(output_dir, f'arcs_{safe_filename}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Arc plot saved: {output_path}")
    
    plt.show()

def plot_arc_statistics(all_arcs, output_dir='output_plots'):
    """
    Creates statistics of detected arcs.
    """
    # Collect statistics
    arc_durations = []
    arc_counts_per_pass = []
    complete_arcs = 0
    total_arcs = 0
    
    for csv_file, file_data in all_arcs.items():
        for tag_id, tag_data in file_data.items():
            arcs = tag_data['arcs']
            
            # Count arcs per pass
            pass_arc_counts = {}
            for arc in arcs:
                pass_id = arc['pass_id']
                if pass_id not in pass_arc_counts:
                    pass_arc_counts[pass_id] = 0
                pass_arc_counts[pass_id] += 1
                
                arc_durations.append(arc['duration'])
                if arc['is_complete_arc']:
                    complete_arcs += 1
                total_arcs += 1
            
            arc_counts_per_pass.extend(pass_arc_counts.values())
    
    # Create figure with statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram of arc durations
    axes[0,0].hist(arc_durations, bins=20, alpha=0.7, edgecolor='black')
    axes[0,0].set_xlabel('Arc duration (s)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title(f'Arc duration distribution\n(Mean: {np.mean(arc_durations):.3f}s)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Number of arcs per pass
    axes[0,1].hist(arc_counts_per_pass, bins=range(1, max(arc_counts_per_pass)+2), 
                   alpha=0.7, edgecolor='black', align='left')
    axes[0,1].set_xlabel('Arcs per pass')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title(f'Arcs per pass distribution\n(Mean: {np.mean(arc_counts_per_pass):.1f})')
    axes[0,1].grid(True, alpha=0.3)
    
    # Complete vs incomplete arcs comparison
    labels = ['Complete Arcs', 'Incomplete Arcs']
    sizes = [complete_arcs, total_arcs - complete_arcs]
    axes[1,0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1,0].set_title('Complete arc proportion')
    
    # Textual statistics
    axes[1,1].axis('off')
    stats_text = f"""=== ARC STATISTICS ===

Total arcs: {total_arcs}
Complete arcs: {complete_arcs} ({100*complete_arcs/total_arcs:.1f}%)
Incomplete arcs: {total_arcs - complete_arcs} ({100*(total_arcs-complete_arcs)/total_arcs:.1f}%)

Arc duration:
  Mean: {np.mean(arc_durations):.3f} s
  Median: {np.median(arc_durations):.3f} s
  Std dev: {np.std(arc_durations):.3f} s
  Min: {np.min(arc_durations):.3f} s
  Max: {np.max(arc_durations):.3f} s

Arcs per pass:
  Mean: {np.mean(arc_counts_per_pass):.1f}
  Median: {np.median(arc_counts_per_pass):.1f}
  Min: {np.min(arc_counts_per_pass)}
  Max: {np.max(arc_counts_per_pass)}"""
    
    axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save summary
    summary_path = os.path.join(output_dir, 'arc_statistics.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Arc statistics saved: {summary_path}")
    
    plt.show()

def main():
    """
    Improved main function that includes arc detection.
    """
    print("=== SEGMENTATION INTO PASSES AND ARCS ===\n")
    
    # Parameters
    abs_threshold = 1
    stat_threshold = 8
    data_folder = "data/test"
    num_interp_points = 200
    smoothing_sigma = 4.0
    
    print(f"Segmentation parameters:")
    print(f"  Absolute threshold: {abs_threshold} s")
    print(f"  Statistical multiplier: {stat_threshold}")
    print(f"  Interpolation points: {num_interp_points}")
    print(f"  Smoothing sigma: {smoothing_sigma}")
    print(f"  Data folder: {data_folder}")
    
    # Step 1: Initial segmentation into passes
    print("\n" + "="*50)
    print("STEP 1: Segmenting into passes...")
    all_segmentations = load_and_segment_csv_files(
        data_folder=data_folder,
        abs_threshold=abs_threshold,
        stat_threshold=stat_threshold
    )
    
    if not all_segmentations:
        print("No data found to process.")
        return
    
    # Step 2: Process passes into arcs
    print("\n" + "="*50)
    print("STEP 2: Detecting arcs within passes...")
    all_arcs = process_segmentation_to_arcs(
        all_segmentations,
        num_interp_points=num_interp_points,
        smoothing_sigma=smoothing_sigma
    )
    
    # Step 3: Visualization
    print("\n" + "="*50)
    print("STEP 3: Generating visualizations...")
    
    for csv_file, file_data in all_arcs.items():
        print(f"Creating arc plot for: {os.path.basename(csv_file)}")
        plot_arcs_for_file(csv_file, file_data)
    
    # Arc statistics
    print("Generating arc statistics...")
    plot_arc_statistics(all_arcs)
    
    print("\n" + "="*50)
    print("Process completed!")
    print("Results saved to 'output_plots'")
    
    return all_arcs

if __name__ == "__main__":
    all_arcs = main()