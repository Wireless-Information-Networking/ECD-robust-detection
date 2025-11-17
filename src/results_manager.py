#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: results_manager.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Results storage and retrieval utilities for RFID data processing workflows.
    Provides comprehensive functions to save/load segmentation results, arc detection
    results, and processing summaries with JSON serialization. Handles metadata
    extraction, numpy array conversion, and robust file operations for analysis
    pipeline persistence and result reproducibility across processing sessions.

License: MIT License
"""

import os
import json
from typing import Any, Dict, Optional
import numpy as np


def save_segmentation_results(all_segmentations: Dict[str, Any], output_path: str = 'output_data/segmentation_results.json'):
    """
    Saves segmentation results to JSON file with metadata extraction.
    
    :param all_segmentations: Dictionary with segmented data from multiple files
    :param output_path: Path to save the results JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {}
    
    for csv_file, file_data in all_segmentations.items():
        serializable_data[csv_file] = {}
        
        for tag_id, tag_data in file_data.items():
            serializable_data[csv_file][tag_id] = {
                'num_segments': tag_data['num_segments'],
                'segments_info': []
            }
            
            # Save segment metadata (not the full data arrays for efficiency)
            for i, segment in enumerate(tag_data['segments']):
                segment_info = {
                    'segment_id': i,
                    'num_samples': len(segment['timestamp']),
                    'duration': float(segment['timestamp'][-1] - segment['timestamp'][0]) if len(segment['timestamp']) > 1 else 0.0,
                    'start_time': float(segment['timestamp'][0]),
                    'end_time': float(segment['timestamp'][-1]),
                    'rssi_range': [float(np.min(segment['rssi'])), float(np.max(segment['rssi']))],
                    'phase_range': [float(np.min(segment['phase'])), float(np.max(segment['phase']))]
                }
                serializable_data[csv_file][tag_id]['segments_info'].append(segment_info)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Segmentation results saved to: {output_path}")

def load_segmentation_results(input_path: str = 'output_data/segmentation_results.json') -> Optional[Dict[str, Any]]:
    """
    Loads segmentation results from JSON file.
    
    :param input_path: Path to load the segmentation results from
    :return: Dictionary with segmentation metadata or None if file doesn't exist
    """
    if os.path.exists(input_path):
        with open(input_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Segmentation results file not found: {input_path}")
        return None


def save_arc_results(all_arcs: Dict[str, Any], output_path: str = 'output_data/arc_results.json'):
    """
    Saves arc detection results to JSON file with comprehensive metadata.
    
    :param all_arcs: Dictionary with arc data from all processed files
    :param output_path: Path to save the arc detection results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert arc data to JSON-serializable format
    serializable_data = {}
    
    for csv_file, file_data in all_arcs.items():
        serializable_data[csv_file] = {}
        
        for tag_id, tag_data in file_data.items():
            arcs_info = []
            
            for arc in tag_data['arcs']:
                arc_info = {
                    'duration': float(arc['duration']),
                    'peak_rssi': float(arc['peak_rssi']),
                    'is_complete_arc': arc['is_complete_arc'],
                    'start_time': float(arc['start_time']),
                    'end_time': float(arc['end_time']),
                    'pass_id': arc['pass_id'],
                    'arc_in_pass': arc['arc_in_pass'],
                    'total_arcs_in_pass': arc['total_arcs_in_pass'],
                    'num_samples': len(arc['timestamp'])
                }
                arcs_info.append(arc_info)
            
            serializable_data[csv_file][tag_id] = {
                'num_arcs': tag_data['num_arcs'],
                'arcs_info': arcs_info
            }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Arc detection results saved to: {output_path}")

def load_arc_results(input_path: str = 'output_data/arc_results.json') -> Optional[Dict[str, Any]]:
    """
    Loads arc detection results from JSON file.
    
    :param input_path: Path to load the arc detection results from
    :return: Dictionary with arc metadata or None if file doesn't exist
    """
    if os.path.exists(input_path):
        with open(input_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Arc detection results file not found: {input_path}")
        return None
