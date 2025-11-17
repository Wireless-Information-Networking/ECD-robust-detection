#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: arc_segmentation.py
Author: Javier del Río
Date: 2025-10-03
Description: 
    Advanced arc segmentation function that combines temporal segmentation,
    interpolation/smoothing, and local minima detection to identify individual
    RFID tag arcs from continuous data streams. Uses a multi-step approach
    for robust arc detection and extraction.

License: MIT License
Dependencies: numpy, scipy, timediff, interpolation
"""

import numpy as np
from typing import List, Dict, Any
import os
import sys

# Import required functions from local modules
from timediff import split_tag_data_by_absolute_and_stat
from interpolation import interpolate_and_smooth_segment, detect_local_minima

def segment_tag_data_into_arcs(
    tag_data: Dict[str, np.ndarray],
    abs_threshold: float = 1.0,
    stat_threshold: float = 2.0,
    num_interp_points: int = 200,
    smoothing_sigma: float = 2.0,
    min_arc_duration: float = 0.1,
    min_arc_samples: int = 5,
    minima_min_distance: int = 10,
    minima_prominence: float = 0.1,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Segments tag data into individual arcs using a multi-step approach:
    1. Split by absolute time difference threshold
    2. Split by statistical time difference (mean + std_threshold * std)
    3. Interpolate and smooth each segment
    4. Find local minima and use them to divide into arcs
    5. Map back to original data timestamps
    
    :param tag_data: Dictionary with 'timestamp', 'rssi', 'phase' arrays
    :param abs_threshold: Absolute time difference threshold (seconds)
    :param stat_threshold: Statistical threshold multiplier for time differences
    :param num_interp_points: Number of points for interpolation
    :param smoothing_sigma: Gaussian smoothing sigma parameter
    :param min_arc_duration: Minimum duration for a valid arc (seconds)
    :param min_arc_samples: Minimum number of samples for a valid arc
    :param minima_min_distance: Minimum distance between local minima
    :param minima_prominence: Prominence threshold for minima detection
    :param verbose: Whether to print progress information
    :return: List of arc dictionaries with original data timestamps
    """
    
    if verbose:
        print(f"Starting arc segmentation with {len(tag_data['timestamp'])} samples")
    
    # Step 1 & 2: Split by absolute and statistical time differences
    if verbose:
        print(f"Step 1-2: Splitting by time differences (abs={abs_threshold}s, stat={stat_threshold})")
    
    segments = split_tag_data_by_absolute_and_stat(
        tag_data, 
        abs_threshold=abs_threshold, 
        stat_threshold=stat_threshold
    )
    
    if verbose:
        print(f"  -> Found {len(segments)} time-based segments")
    
    if not segments:
        if verbose:
            print("  -> No segments found, returning empty list")
        return []
    
    all_arcs = []
    
    # Process each segment
    for seg_idx, segment in enumerate(segments):
        if verbose:
            print(f"\nProcessing segment {seg_idx + 1}/{len(segments)} ({len(segment['timestamp'])} samples)")
        
        # Skip segments that are too small
        if len(segment['timestamp']) < min_arc_samples:
            if verbose:
                print(f"  -> Skipping segment {seg_idx + 1}: too few samples ({len(segment['timestamp'])} < {min_arc_samples})")
            continue
        
        # Step 3: Interpolate and smooth the segment
        if verbose:
            print(f"  Step 3: Interpolating and smoothing (points={num_interp_points}, sigma={smoothing_sigma})")
        
        try:
            interpolated_segment = interpolate_and_smooth_segment(
                segment,
                num_points=num_interp_points,
                smoothing_method='gaussian',
                smoothing_params={'sigma': smoothing_sigma}
            )
        except Exception as e:
            if verbose:
                print(f"  -> Error interpolating segment {seg_idx + 1}: {e}")
            continue
        
        # Step 4: Find local minima in the smoothed RSSI signal
        if verbose:
            print(f"  Step 4: Finding local minima (min_distance={minima_min_distance}, prominence={minima_prominence})")
        
        try:
            minima_indices = detect_local_minima(
                interpolated_segment['rssi_smooth'],
                min_distance=minima_min_distance,
                prominence_threshold=minima_prominence
            )
        except Exception as e:
            if verbose:
                print(f"  -> Error detecting minima in segment {seg_idx + 1}: {e}")
            # If minima detection fails, treat entire segment as one arc
            minima_indices = np.array([])
        
        if verbose:
            print(f"  -> Found {len(minima_indices)} minima in interpolated data")
        
        # Step 5: Map minima back to original data and create arcs
        arcs_from_segment = _create_arcs_from_minima(
            segment, 
            interpolated_segment, 
            minima_indices,
            min_arc_duration=min_arc_duration,
            min_arc_samples=min_arc_samples,
            verbose=verbose
        )
        
        # Add segment context to each arc
        for arc in arcs_from_segment:
            arc['source_segment'] = seg_idx
            arc['total_segments'] = len(segments)
        
        all_arcs.extend(arcs_from_segment)
        
        if verbose:
            print(f"  -> Extracted {len(arcs_from_segment)} arcs from segment {seg_idx + 1}")
    
    # CORRECCIÓN: Validar estructura de arcos antes de devolver
    all_arcs = _validate_arc_structure(all_arcs)
    
    if verbose:
        print(f"\n=== ARC SEGMENTATION COMPLETED ===")
        print(f"Total arcs found: {len(all_arcs)}")
        if all_arcs:
            durations = [arc['duration'] for arc in all_arcs]
            print(f"Arc durations: mean={np.mean(durations):.3f}s, std={np.std(durations):.3f}s")
            print(f"Duration range: {np.min(durations):.3f}s to {np.max(durations):.3f}s")
    
    return all_arcs

def _create_arcs_from_minima(
    original_segment: Dict[str, np.ndarray],
    interpolated_segment: Dict[str, np.ndarray],
    minima_indices: np.ndarray,
    min_arc_duration: float = 0.1,
    min_arc_samples: int = 5,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Creates arc segments from original data using minima found in interpolated data.
    
    :param original_segment: Original segment data
    :param interpolated_segment: Interpolated and smoothed segment data
    :param minima_indices: Indices of minima in interpolated data
    :param min_arc_duration: Minimum arc duration
    :param min_arc_samples: Minimum arc samples
    :param verbose: Verbose output
    :return: List of arc dictionaries
    """
    
    orig_timestamps = original_segment['timestamp']
    orig_rssi = original_segment['rssi']
    orig_phase = original_segment['phase']
    
    interp_timestamps = interpolated_segment['timestamp']
    
    # If no minima found, return entire segment as one arc
    if len(minima_indices) == 0:
        if verbose:
            print("    No minima found, treating entire segment as one arc")
        
        duration = orig_timestamps[-1] - orig_timestamps[0]
        if duration >= min_arc_duration and len(orig_timestamps) >= min_arc_samples:
            peak_idx = np.argmax(orig_rssi)
            peak_rssi = orig_rssi[peak_idx]
            peak_time = orig_timestamps[peak_idx]
            
            # CORRECCIÓN: Asegurar que todos los campos estén presentes
            return [{
                'timestamp': orig_timestamps,
                'rssi': orig_rssi,
                'phase': orig_phase,
                'duration': duration,
                'num_samples': len(orig_timestamps),
                'arc_type': 'complete_segment',
                'has_clear_minima': False,
                'start_time': orig_timestamps[0],
                'end_time': orig_timestamps[-1],
                'peak_rssi': peak_rssi,
                'peak_time': peak_time,
                'peak_index': peak_idx,
                'start_index_in_segment': 0,
                'end_index_in_segment': len(orig_timestamps),
                'rssi_range': np.max(orig_rssi) - np.min(orig_rssi),
                'mean_rssi': np.mean(orig_rssi),
                'std_rssi': np.std(orig_rssi)
            }]
        else:
            return []
    
    # Map minima from interpolated data back to original data timestamps  
    minima_times = interp_timestamps[minima_indices]
    
    if verbose:
        print(f"    Mapping {len(minima_times)} minima back to original timestamps")
    
    # Find closest original timestamps to minima times
    original_split_indices = []
    for minima_time in minima_times:
        # Find the original timestamp closest to (but not after) the minima time
        valid_indices = np.where(orig_timestamps <= minima_time)[0]
        if len(valid_indices) > 0:
            closest_idx = valid_indices[-1]  # Last valid index (closest but not after)
            original_split_indices.append(closest_idx)
    
    # Remove duplicates and sort
    original_split_indices = sorted(set(original_split_indices))
    
    # Add start and end indices
    split_points = [0] + original_split_indices + [len(orig_timestamps)]
    split_points = sorted(set(split_points))  # Remove duplicates
    
    if verbose:
        print(f"    Split points in original data: {split_points}")
    
    # Create arcs from split points
    arcs = []
    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]
        
        # Skip if segment is too small
        if end_idx - start_idx < min_arc_samples:
            if verbose:
                print(f"    Skipping arc {i+1}: too few samples ({end_idx - start_idx} < {min_arc_samples})")
            continue
        
        arc_timestamps = orig_timestamps[start_idx:end_idx]
        arc_rssi = orig_rssi[start_idx:end_idx]
        arc_phase = orig_phase[start_idx:end_idx]
        
        duration = arc_timestamps[-1] - arc_timestamps[0]
        
        # Skip if duration is too short
        if duration < min_arc_duration:
            if verbose:
                print(f"    Skipping arc {i+1}: duration too short ({duration:.3f}s < {min_arc_duration}s)")
            continue
        
        # Determine arc characteristics
        peak_idx = np.argmax(arc_rssi)
        peak_rssi = arc_rssi[peak_idx]
        peak_time = arc_timestamps[peak_idx]
        
        # Check if arc has clear boundaries (starts and ends with minima)
        has_clear_start = start_idx > 0 and start_idx in original_split_indices
        has_clear_end = end_idx < len(orig_timestamps) and (end_idx - 1) in original_split_indices
        has_clear_minima = has_clear_start or has_clear_end
        
        # Determine arc type
        if has_clear_start and has_clear_end:
            arc_type = 'complete_arc'
        elif has_clear_start or has_clear_end:
            arc_type = 'partial_arc'
        else:
            arc_type = 'segment_boundary'
        
        # CORRECCIÓN: Asegurar que todos los campos estén presentes y sean consistentes
        arc_dict = {
            'timestamp': arc_timestamps,
            'rssi': arc_rssi,
            'phase': arc_phase,
            'duration': duration,
            'num_samples': len(arc_timestamps),
            'arc_type': arc_type,
            'has_clear_minima': has_clear_minima,
            'start_time': arc_timestamps[0],
            'end_time': arc_timestamps[-1],
            'peak_rssi': peak_rssi,
            'peak_time': peak_time,
            'peak_index': peak_idx,
            'start_index_in_segment': start_idx,
            'end_index_in_segment': end_idx,
            'rssi_range': np.max(arc_rssi) - np.min(arc_rssi),
            'mean_rssi': np.mean(arc_rssi),
            'std_rssi': np.std(arc_rssi)
        }
        
        arcs.append(arc_dict)
        
        if verbose:
            print(f"    Arc {len(arcs)}: {duration:.3f}s, {len(arc_timestamps)} samples, type={arc_type}")
    
    return arcs

def _validate_arc_structure(arcs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validates and ensures all arcs have the required fields.
    
    :param arcs: List of arc dictionaries
    :return: List of validated arc dictionaries
    """
    required_fields = [
        'timestamp', 'rssi', 'phase', 'duration', 'num_samples',
        'arc_type', 'has_clear_minima', 'start_time', 'end_time',
        'peak_rssi', 'peak_time', 'peak_index', 'start_index_in_segment',
        'end_index_in_segment', 'rssi_range', 'mean_rssi', 'std_rssi'
    ]
    
    validated_arcs = []
    
    for i, arc in enumerate(arcs):
        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in arc]
        
        if missing_fields:
            print(f"Warning: Arc {i+1} missing fields: {missing_fields}")
            
            # Try to compute missing fields if possible
            if 'rssi' in arc and 'timestamp' in arc:
                if 'rssi_range' not in arc:
                    arc['rssi_range'] = np.max(arc['rssi']) - np.min(arc['rssi'])
                if 'mean_rssi' not in arc:
                    arc['mean_rssi'] = np.mean(arc['rssi'])
                if 'std_rssi' not in arc:
                    arc['std_rssi'] = np.std(arc['rssi'])
                if 'peak_rssi' not in arc:
                    peak_idx = np.argmax(arc['rssi'])
                    arc['peak_rssi'] = arc['rssi'][peak_idx]
                    arc['peak_time'] = arc['timestamp'][peak_idx]
                    arc['peak_index'] = peak_idx
                if 'duration' not in arc:
                    arc['duration'] = arc['timestamp'][-1] - arc['timestamp'][0]
                if 'num_samples' not in arc:
                    arc['num_samples'] = len(arc['timestamp'])
                if 'start_time' not in arc:
                    arc['start_time'] = arc['timestamp'][0]
                if 'end_time' not in arc:
                    arc['end_time'] = arc['timestamp'][-1]
                
                # Set default values for missing metadata
                if 'arc_type' not in arc:
                    arc['arc_type'] = 'unknown'
                if 'has_clear_minima' not in arc:
                    arc['has_clear_minima'] = False
                if 'start_index_in_segment' not in arc:
                    arc['start_index_in_segment'] = 0
                if 'end_index_in_segment' not in arc:
                    arc['end_index_in_segment'] = len(arc['timestamp'])
        
        validated_arcs.append(arc)
    
    return validated_arcs

def plot_arc_segmentation_results(
    original_data: Dict[str, np.ndarray],
    arcs: List[Dict[str, Any]],
    tag_id: str = "Unknown",
    save_plot: bool = True,
    output_dir: str = "output_plots"
) -> None:
    """
    Plots the results of arc segmentation showing original data and detected arcs.
    
    :param original_data: Original tag data
    :param arcs: List of detected arcs
    :param tag_id: Tag identifier for plot title
    :param save_plot: Whether to save the plot
    :param output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    
    if not arcs:
        print("No arcs to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot original data
    ax1.plot(original_data['timestamp'], original_data['rssi'], 
             'k-', alpha=0.3, linewidth=0.5, label='Original RSSI', zorder=1)
    
    # Plot each arc with different colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, arc in enumerate(arcs):
        color = colors[i % len(colors)]
        
        # Plot arc RSSI
        ax1.plot(arc['timestamp'], arc['rssi'], 
                color=color, linewidth=2, alpha=0.8, 
                label=f'Arc {i+1} ({arc["arc_type"]}, {arc["duration"]:.2f}s)', zorder=2)
        
        # Mark peak
        peak_idx = arc['peak_index']
        ax1.plot(arc['timestamp'][peak_idx], arc['rssi'][peak_idx], 
                '*', color=color, markersize=12, zorder=3)
        
        # Mark boundaries if clear minima
        if arc['has_clear_minima']:
            ax1.axvline(x=arc['start_time'], color=color, linestyle='--', alpha=0.7, zorder=1)
            ax1.axvline(x=arc['end_time'], color=color, linestyle='--', alpha=0.7, zorder=1)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('RSSI (dBm)')
    ax1.set_title(f'Tag {tag_id} - Arc Segmentation Results ({len(arcs)} arcs detected)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot phase data
    ax2.plot(original_data['timestamp'], original_data['phase'], 
             'k-', alpha=0.3, linewidth=0.5, label='Original Phase', zorder=1)
    
    for i, arc in enumerate(arcs):
        color = colors[i % len(colors)]
        ax2.plot(arc['timestamp'], arc['phase'], 
                color=color, linewidth=2, alpha=0.8, zorder=2)
        
        # Mark boundaries
        if arc['has_clear_minima']:
            ax2.axvline(x=arc['start_time'], color=color, linestyle='--', alpha=0.7, zorder=1)
            ax2.axvline(x=arc['end_time'], color=color, linestyle='--', alpha=0.7, zorder=1)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title(f'Tag {tag_id} - Phase Data with Arc Boundaries')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        filename = f'arc_segmentation_{tag_id}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {filepath}")
    
    plt.show()

def analyze_arc_statistics(arcs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes statistics of detected arcs.
    
    :param arcs: List of detected arcs
    :return: Dictionary with arc statistics
    """
    if not arcs:
        return {'num_arcs': 0}
    
    # CORRECCIÓN: Usar get() con valores por defecto para campos que podrían faltar
    durations = [arc.get('duration', 0) for arc in arcs]
    num_samples = [arc.get('num_samples', 0) for arc in arcs]
    peak_rssi_values = [arc.get('peak_rssi', 0) for arc in arcs]
    
    # Calcular rssi_range con manejo de errores
    rssi_ranges = []
    for arc in arcs:
        if 'rssi_range' in arc:
            rssi_ranges.append(arc['rssi_range'])
        elif 'rssi' in arc and len(arc['rssi']) > 0:
            rssi_ranges.append(np.max(arc['rssi']) - np.min(arc['rssi']))
        else:
            rssi_ranges.append(0)
    
    # Count arc types
    arc_types = {}
    for arc in arcs:
        arc_type = arc.get('arc_type', 'unknown')
        arc_types[arc_type] = arc_types.get(arc_type, 0) + 1
    
    # Count arcs with clear minima
    clear_minima_count = sum(1 for arc in arcs if arc.get('has_clear_minima', False))
    
    # Filtrar valores válidos para estadísticas
    valid_durations = [d for d in durations if d > 0]
    valid_samples = [s for s in num_samples if s > 0]
    valid_peak_rssi = [p for p in peak_rssi_values if p != 0]
    valid_rssi_ranges = [r for r in rssi_ranges if r > 0]
    
    stats = {
        'num_arcs': len(arcs),
        'duration_stats': {
            'mean': np.mean(valid_durations) if valid_durations else 0,
            'std': np.std(valid_durations) if valid_durations else 0,
            'min': np.min(valid_durations) if valid_durations else 0,
            'max': np.max(valid_durations) if valid_durations else 0,
            'median': np.median(valid_durations) if valid_durations else 0
        },
        'samples_stats': {
            'mean': np.mean(valid_samples) if valid_samples else 0,
            'std': np.std(valid_samples) if valid_samples else 0,
            'min': np.min(valid_samples) if valid_samples else 0,
            'max': np.max(valid_samples) if valid_samples else 0,
            'median': np.median(valid_samples) if valid_samples else 0
        },
        'peak_rssi_stats': {
            'mean': np.mean(valid_peak_rssi) if valid_peak_rssi else 0,
            'std': np.std(valid_peak_rssi) if valid_peak_rssi else 0,
            'min': np.min(valid_peak_rssi) if valid_peak_rssi else 0,
            'max': np.max(valid_peak_rssi) if valid_peak_rssi else 0,
            'median': np.median(valid_peak_rssi) if valid_peak_rssi else 0
        },
        'rssi_range_stats': {
            'mean': np.mean(valid_rssi_ranges) if valid_rssi_ranges else 0,
            'std': np.std(valid_rssi_ranges) if valid_rssi_ranges else 0,
            'min': np.min(valid_rssi_ranges) if valid_rssi_ranges else 0,
            'max': np.max(valid_rssi_ranges) if valid_rssi_ranges else 0,
            'median': np.median(valid_rssi_ranges) if valid_rssi_ranges else 0
        },
        'arc_types': arc_types,
        'clear_minima_count': clear_minima_count,
        'clear_minima_percentage': (clear_minima_count / len(arcs)) * 100 if arcs else 0
    }
    
    return stats

def print_arc_statistics(stats: Dict[str, Any]) -> None:
    """
    Prints arc statistics in a formatted way.
    
    :param stats: Arc statistics dictionary
    """
    print("\n" + "="*50)
    print("ARC SEGMENTATION STATISTICS")
    print("="*50)
    
    if stats['num_arcs'] == 0:
        print("No arcs detected.")
        return
    
    print(f"Total arcs detected: {stats['num_arcs']}")
    print(f"Arcs with clear minima: {stats['clear_minima_count']} ({stats['clear_minima_percentage']:.1f}%)")
    
    print(f"\nDuration statistics:")
    d_stats = stats['duration_stats']
    print(f"  Mean: {d_stats['mean']:.3f}s ± {d_stats['std']:.3f}s")
    print(f"  Range: {d_stats['min']:.3f}s to {d_stats['max']:.3f}s")
    print(f"  Median: {d_stats['median']:.3f}s")
    
    print(f"\nSample count statistics:")
    s_stats = stats['samples_stats']
    print(f"  Mean: {s_stats['mean']:.1f} ± {s_stats['std']:.1f}")
    print(f"  Range: {s_stats['min']} to {s_stats['max']}")
    print(f"  Median: {s_stats['median']:.1f}")
    
    print(f"\nPeak RSSI statistics:")
    p_stats = stats['peak_rssi_stats']
    print(f"  Mean: {p_stats['mean']:.2f} dBm ± {p_stats['std']:.2f} dBm")
    print(f"  Range: {p_stats['min']:.2f} dBm to {p_stats['max']:.2f} dBm")
    print(f"  Median: {p_stats['median']:.2f} dBm")
    
    print(f"\nRSSI range statistics:")
    r_stats = stats['rssi_range_stats']
    print(f"  Mean: {r_stats['mean']:.2f} dBm ± {r_stats['std']:.2f} dBm")
    print(f"  Range: {r_stats['min']:.2f} dBm to {r_stats['max']:.2f} dBm")
    print(f"  Median: {r_stats['median']:.2f} dBm")
    
    print(f"\nArc types:")
    for arc_type, count in stats['arc_types'].items():
        percentage = (count / stats['num_arcs']) * 100
        print(f"  {arc_type}: {count} ({percentage:.1f}%)")

# Example usage and testing
if __name__ == "__main__":
    # Import data loading function
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from csv_data_loader import extract_tag_data
    from config_manager import load_optimized_parameters
    
    print("=== ARC SEGMENTATION EXAMPLE ===")
    
    # Load test data
    csv_file = 'data/2025-10-07/static-espurio-1.csv'
    print(f"Loading data from {csv_file}...")
    
    try:
        tag_data_dict = extract_tag_data(csv_file)
        if not tag_data_dict:
            print("Could not load data.")
            exit()
        
        # Select first tag
        tag_id = list(tag_data_dict.keys())[0]
        tag_data = tag_data_dict[tag_id]
        
        # Convert to numpy arrays
        for key in ['timestamp', 'rssi', 'phase']:
            tag_data[key] = np.array(tag_data[key])
        
        print(f"Processing tag {tag_id} with {len(tag_data['timestamp'])} samples")
        
        # Load optimized parameters using the config_manager function
        optimized_params = load_optimized_parameters('output_data/optimized_parameters.json')
        
        # Create configuration from loaded parameters
        config = {
            'name': 'Optimized' if 'optimization_date' in optimized_params else 'Default',
            'abs_threshold': optimized_params.get('abs_threshold', 1.0),
            'stat_threshold': optimized_params.get('stat_threshold', 2.0),
            'num_interp_points': int(optimized_params.get('num_interp_points', 200)),
            'smoothing_sigma': optimized_params.get('smoothing_sigma', 2.0),
            'minima_prominence': optimized_params.get('minima_prominence', 0.1),
            'min_arc_duration': optimized_params.get('min_arc_duration', 0.1),
            'min_arc_samples': optimized_params.get('min_arc_samples', 5),
            'minima_min_distance': optimized_params.get('minima_min_distance', 10)
        }
        
        print(f"\n{'='*20} {config['name']} Configuration {'='*20}")
        print("Parameters loaded:")
        for key, value in config.items():
            if key != 'name':
                print(f"  {key}: {value}")
        
        # Use the loaded parameters in the segmentation function
        arcs = segment_tag_data_into_arcs(
            tag_data,
            abs_threshold=config['abs_threshold'],
            stat_threshold=config['stat_threshold'],
            num_interp_points=config['num_interp_points'],
            smoothing_sigma=config['smoothing_sigma'],
            min_arc_duration=config['min_arc_duration'],
            min_arc_samples=config['min_arc_samples'],
            minima_min_distance=config['minima_min_distance'],
            minima_prominence=config['minima_prominence'],
            verbose=True
        )
        
        # Analyze and print statistics
        stats = analyze_arc_statistics(arcs)
        print_arc_statistics(stats)
        
        # Plot results
        plot_arc_segmentation_results(
            tag_data, 
            arcs, 
            f"{tag_id}_{config['name'].lower()}", 
            save_plot=True
        )
        
        # Show summary
        print(f"\n{'='*50}")
        print("ARC SEGMENTATION SUMMARY")
        print("="*50)
        print(f"Configuration: {config['name']}")
        print(f"Total arcs detected: {len(arcs)}")
        
        if 'optimization_date' in optimized_params:
            print(f"Parameters optimized on: {optimized_params['optimization_date']}")
        
        print("\nParameters used:")
        for key, value in config.items():
            if key != 'name':
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()