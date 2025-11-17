#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: ejecutar_segmentacion.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Script for executing RFID tag segmentation into arcs and saving results in multiple formats.
    Provides functionality to save, load, and analyze segmentation results with comprehensive
    data persistence and statistical analysis capabilities.

License: MIT License
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
from datetime import datetime
from test_div import main as run_segmentation

def save_arc_results(all_arcs, output_dir='output_data'):
    """
    Saves arc segmentation results in different formats.
    
    :param all_arcs: Dictionary with all detected arcs
    :param output_dir: Directory where to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for unique files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save complete data in pickle format (preserves numpy structures)
    pickle_path = os.path.join(output_dir, f'arc_data_{timestamp}.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_arcs, f)
    print(f"Complete data saved to: {pickle_path}")
    
    # 2. Save summary in JSON format (human readable)
    json_summary = create_json_summary(all_arcs)
    json_path = os.path.join(output_dir, f'arc_summary_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"JSON summary saved to: {json_path}")
    
    # 3. Save arc data in CSV format
    csv_path = os.path.join(output_dir, f'arc_features_{timestamp}.csv')
    save_arc_features_csv(all_arcs, csv_path)
    print(f"Arc features saved to: {csv_path}")
    
    # 4. Save configuration used
    config_path = os.path.join(output_dir, f'config_{timestamp}.json')
    save_config(config_path)
    print(f"Configuration saved to: {config_path}")
    
    return {
        'pickle_path': pickle_path,
        'json_path': json_path,
        'csv_path': csv_path,
        'config_path': config_path
    }

def create_json_summary(all_arcs):
    """Creates a summary in JSON serializable format."""
    summary = {
        'metadata': {
            'creation_time': datetime.now().isoformat(),
            'total_files': len(all_arcs),
            'processing_info': 'Arc segmentation results'
        },
        'files': {}
    }
    
    for csv_file, file_data in all_arcs.items():
        file_name = os.path.basename(csv_file)
        file_summary = {
            'tags': {},
            'total_arcs': 0
        }
        
        for tag_id, tag_data in file_data.items():
            arcs = tag_data['arcs']
            
            # Statistics per tag
            tag_stats = {
                'num_arcs': len(arcs),
                'arc_durations': [float(arc['duration']) for arc in arcs],
                'complete_arcs': sum(1 for arc in arcs if arc['is_complete_arc']),
                'passes': list(set(arc['pass_id'] for arc in arcs)),
                'arc_peaks': [float(arc['peak_rssi']) for arc in arcs]
            }
            
            file_summary['tags'][tag_id] = tag_stats
            file_summary['total_arcs'] += len(arcs)
        
        summary['files'][file_name] = file_summary
    
    return summary

def save_arc_features_csv(all_arcs, csv_path):
    """Saves arc features for each arc in CSV format."""
    import pandas as pd
    
    arc_records = []
    
    for csv_file, file_data in all_arcs.items():
        file_name = os.path.basename(csv_file)
        
        for tag_id, tag_data in file_data.items():
            arcs = tag_data['arcs']
            
            for arc_idx, arc in enumerate(arcs):
                record = {
                    'file_name': file_name,
                    'tag_id': tag_id,
                    'arc_id': arc_idx,
                    'pass_id': arc['pass_id'],
                    'arc_in_pass': arc['arc_in_pass'],
                    'total_arcs_in_pass': arc['total_arcs_in_pass'],
                    'duration': arc['duration'],
                    'start_time': arc['start_time'],
                    'end_time': arc['end_time'],
                    'peak_rssi': arc['peak_rssi'],
                    'is_complete_arc': arc['is_complete_arc'],
                    'num_samples': len(arc['timestamp']),
                    'rssi_mean': np.mean(arc['rssi']),
                    'rssi_std': np.std(arc['rssi']),
                    'rssi_min': np.min(arc['rssi']),
                    'rssi_max': np.max(arc['rssi']),
                    'phase_mean': np.mean(arc['phase']),
                    'phase_std': np.std(arc['phase'])
                }
                arc_records.append(record)
    
    df = pd.DataFrame(arc_records)
    df.to_csv(csv_path, index=False)

def save_config(config_path):
    """Saves the configuration used in processing."""
    config = {
        'segmentation_params': {
            'abs_threshold': 0.5,
            'stat_threshold': 2,
            'num_interp_points': 100,
            'smoothing_sigma': 3.0
        },
        'arc_detection_params': {
            'min_arc_duration': 0.1,
            'min_distance': 10,
            'prominence_threshold': 0.1
        },
        'data_source': 'data/test',
        'processing_date': datetime.now().isoformat()
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_arc_results(pickle_path):
    """Loads previously saved results."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def analyze_saved_results(pickle_path):
    """Analyzes saved results and shows statistics."""
    all_arcs = load_arc_results(pickle_path)
    
    print("=== ANALYSIS OF SAVED RESULTS ===")
    print(f"Data loaded from: {pickle_path}")
    
    total_files = len(all_arcs)
    total_arcs = 0
    complete_arcs = 0
    all_durations = []
    
    for csv_file, file_data in all_arcs.items():
        file_name = os.path.basename(csv_file)
        print(f"\nFile: {file_name}")
        
        for tag_id, tag_data in file_data.items():
            arcs = tag_data['arcs']
            tag_complete = sum(1 for arc in arcs if arc['is_complete_arc'])
            tag_durations = [arc['duration'] for arc in arcs]
            
            print(f"  Tag {tag_id}: {len(arcs)} arcs ({tag_complete} complete)")
            print(f"    Mean duration: {np.mean(tag_durations):.3f}s")
            
            total_arcs += len(arcs)
            complete_arcs += tag_complete
            all_durations.extend(tag_durations)
    
    print(f"\n=== GLOBAL SUMMARY ===")
    print(f"Files processed: {total_files}")
    print(f"Total arcs: {total_arcs}")
    print(f"Complete arcs: {complete_arcs} ({100*complete_arcs/total_arcs:.1f}%)")
    print(f"Mean arc duration: {np.mean(all_durations):.3f}s ± {np.std(all_durations):.3f}s")

def execute_and_save():
    """
    Executes segmentation and automatically saves results.
    """
    print("=== EXECUTING SEGMENTATION AND SAVING RESULTS ===\n")
    
    # Execute segmentation
    print("Starting segmentation process...")
    all_arcs = run_segmentation()
    
    if not all_arcs:
        print("No results obtained from segmentation.")
        return None
    
    # Save results
    print("\nSaving results...")
    saved_paths = save_arc_results(all_arcs)
    
    # Show summary
    print("\n=== SUMMARY OF SAVED FILES ===")
    for description, path in saved_paths.items():
        print(f"{description}: {path}")
    
    # Quick analysis
    print("\n=== QUICK ANALYSIS ===")
    analyze_saved_results(saved_paths['pickle_path'])
    
    return saved_paths

def main():
    """Main function that allows different operation modes."""
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == 'analyze' and len(sys.argv) > 2:
            # Analyze existing results
            pickle_path = sys.argv[2]
            analyze_saved_results(pickle_path)
        elif mode == 'load' and len(sys.argv) > 2:
            # Load and show results
            pickle_path = sys.argv[2]
            all_arcs = load_arc_results(pickle_path)
            print(f"Loaded data from {len(all_arcs)} files")
            return all_arcs
        else:
            print("Usage:")
            print("  python ejecutar_segmentacion.py                    # Execute and save")
            print("  python ejecutar_segmentacion.py analyze <path>     # Analyze results")
            print("  python ejecutar_segmentacion.py load <path>        # Load results")
    else:
        # Default mode: execute and save
        return execute_and_save()

if __name__ == "__main__":
    results = main()