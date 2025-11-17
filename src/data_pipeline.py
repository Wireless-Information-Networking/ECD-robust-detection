#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: data_pipeline.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    High-level data processing pipeline for RFID tag analysis workflows.
    Orchestrates CSV file loading, data extraction, and segmentation operations
    across multiple files and tags. Provides batch processing capabilities
    with comprehensive error handling and progress reporting for large-scale
    RFID data analysis and automated processing workflows.

License: MIT License
"""


import os
from typing import Dict, Any

from csv_data_loader import extract_tag_data
from timediff import split_tag_data_by_absolute_and_stat

def load_and_segment_csv_files(data_folder: str = 'data', abs_threshold: float = 1.0, stat_threshold: float = 1.0) -> Dict[str, Any]:
    """
    Loads multiple CSV files and segments data based on absolute time differences.
    
    :param data_folder: Folder containing CSV files to process
    :param abs_threshold: Absolute time difference threshold for segmentation
    :param stat_threshold: Statistical threshold multiplier for segmentation
    :return: Dictionary with segmented data organized by file and tag
    """    
    all_segmentations = {}
    
    # Search for all CSV files in folder recursively
    csv_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        
        try:
            # Extract tag data from CSV file
            tag_data = extract_tag_data(csv_file)
            
            if not tag_data:
                print(f"  Could not extract data from {csv_file}")
                continue
            
            file_segments = {}
            
            # Process segmentation for each tag in the file
            for tag_id, tag_values in tag_data.items():
                print(f"  Tag {tag_id}: {len(tag_values['timestamp'])} samples")
                
                # Apply time-based segmentation algorithm
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
