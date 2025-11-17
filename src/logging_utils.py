#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: logging_utils.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Logging and summary utilities for RFID data processing workflows.
    Provides comprehensive functions to save processing logs, create data summaries,
    and manage analysis metadata with JSON serialization. Supports workflow tracking,
    statistical summaries of tag datasets, and processing history management for 
    debugging and analysis reproducibility across multiple processing sessions.

License: MIT License
"""

from typing import Dict, Any, List
import os
import json
import numpy as np

def save_processing_log(log_entries: List[str], output_path: str = 'output_data/processing_log.txt'):
    """
    Saves processing log entries to text file with timestamp header.
    
    :param log_entries: List of log entries to save
    :param output_path: Path to save the log file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"Processing Log - {np.datetime64('now')}\n")
        f.write("=" * 50 + "\n\n")
        
        for entry in log_entries:
            f.write(f"{entry}\n")
    
    print(f"Processing log saved to: {output_path}")

def create_data_summary(all_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Creates a comprehensive summary of all loaded RFID data.
    
    :param all_data: Dictionary with data from multiple files
    :return: Summary dictionary with statistics and metadata
    """
    summary = {
        'total_files': len(all_data),
        'total_tags': 0,
        'total_samples': 0,
        'files_info': {},
        'tag_ids': set(),
        'time_range': {'min': float('inf'), 'max': float('-inf')}
    }
    
    for csv_file, file_data in all_data.items():
        file_info = {
            'num_tags': len(file_data),
            'tags': list(file_data.keys()),
            'total_samples': 0
        }
        
        for tag_id, tag_data in file_data.items():
            summary['tag_ids'].add(tag_id)
            num_samples = len(tag_data['timestamp'])
            file_info['total_samples'] += num_samples
            summary['total_samples'] += num_samples
            
            # Update global time range across all data
            if len(tag_data['timestamp']) > 0:
                min_time = np.min(tag_data['timestamp'])
                max_time = np.max(tag_data['timestamp'])
                summary['time_range']['min'] = min(summary['time_range']['min'], min_time)
                summary['time_range']['max'] = max(summary['time_range']['max'], max_time)
        
        summary['files_info'][csv_file] = file_info
        summary['total_tags'] += file_info['num_tags']
    
    summary['unique_tags'] = len(summary['tag_ids'])
    summary['tag_ids'] = list(summary['tag_ids'])
    
    return summary

def save_data_summary(summary: Dict[str, Any], output_path: str = 'output_data/data_summary.json'):
    """
    Saves data summary to JSON file with proper serialization handling.
    
    :param summary: Summary dictionary to save
    :param output_path: Path to save the summary JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert sets to lists for JSON serialization compatibility
    serializable_summary = summary.copy()
    if 'tag_ids' in serializable_summary and isinstance(serializable_summary['tag_ids'], set):
        serializable_summary['tag_ids'] = list(serializable_summary['tag_ids'])
    
    with open(output_path, 'w') as f:
        json.dump(serializable_summary, f, indent=2, default=str)
    
    print(f"Data summary saved to: {output_path}")
