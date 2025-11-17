#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: csv_data_loader.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    CSV data loading and extraction utilities for RFID tag data processing.
    Provides specialized functions for reading, writing and processing RFID tag 
    data from CSV files with robust data cleaning, timestamp normalization, and
    batch processing capabilities. Handles multiple file imports, directory
    scanning, and data export operations for comprehensive RFID analysis workflows.

License: MIT License
"""

import pandas as pd
import os
import numpy as np
from typing import Dict, List, Any


def extract_tag_data(csv_path):
    """
    Extracts RFID tag data from CSV file and organizes it by tag ID.
    
    :param csv_path: Path to the CSV file containing RFID data
    :return: Dictionary with tag IDs as keys and tag data as values
    """
    df = pd.read_csv(csv_path)
    
    # Clean whitespace in column names if necessary
    df.columns = df.columns.str.strip()
    
    # Clean idHex from special characters (spaces, tabs, line breaks, etc.)
    df['idHex'] = df['idHex'].astype(str).str.replace(r'\s+', '', regex=True)
    
    # Select relevant columns for RFID analysis
    cols = ['idHex', 'peakRssi', 'phase', 'time_reader']
    df = df[cols]
    
    # Group by idHex and extract data into numpy arrays for efficient processing
    tag_dict = {}
    for tag_id, group in df.groupby('idHex'):
        tag_dict[tag_id] = {
            'rssi': group['peakRssi'].to_numpy(),
            'phase': group['phase'].to_numpy(),
            'timestamp': group['time_reader'].to_numpy()
        }
    return tag_dict

def normalize_timestamps(tag_data):
    """
    Normalizes timestamps to start from zero across all tags.
    
    :param tag_data: Dictionary containing tag data
    :return: Dictionary with normalized timestamps
    """
    # Find the minimum timestamp among all tags
    min_time = min(np.min(values['timestamp']) for values in tag_data.values())
    
    # Subtract minimum from all timestamps to start from zero
    for values in tag_data.values():
        values['timestamp'] = values['timestamp'] - min_time
    
    return tag_data

def export_data_to_csv(tag_data: Dict[str, Any], output_path: str):
    """
    Exports tag data to CSV format with proper structure.
    
    :param tag_data: Dictionary with tag data
    :param output_path: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Combine all tag data into a single DataFrame
    all_data = []
    
    for tag_id, data in tag_data.items():
        df_tag = pd.DataFrame({
            'idHex': [tag_id] * len(data['timestamp']),
            'peakRssi': data['rssi'],
            'phase': data['phase'],
            'time_reader': data['timestamp']
        })
        all_data.append(df_tag)
    
    # Concatenate all data into unified structure
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp for chronological order
    combined_df = combined_df.sort_values('time_reader')
    
    # Save to CSV file
    combined_df.to_csv(output_path, index=False)
    print(f"Data exported to CSV: {output_path}")

def import_data_from_multiple_csvs(csv_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Imports data from multiple CSV files with error handling.
    
    :param csv_files: List of CSV file paths
    :return: Dictionary with data from all successfully loaded files
    """
    all_data = {}
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"Loading data from: {csv_file}")
            tag_data = extract_tag_data(csv_file)
            if tag_data:
                all_data[csv_file] = tag_data
                print(f"  Loaded {len(tag_data)} tags")
            else:
                print(f"  No data found in {csv_file}")
        else:
            print(f"  File not found: {csv_file}")
    
    return all_data

def get_csv_files_from_directory(directory: str, recursive: bool = True) -> List[str]:
    """
    Gets all CSV files from a directory with optional recursive search.
    
    :param directory: Directory to search for CSV files
    :param recursive: Whether to search recursively in subdirectories
    :return: List of CSV file paths sorted alphabetically
    """
    csv_files = []
    
    if recursive:
        # Search recursively through all subdirectories
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
    else:
        # Search only in the specified directory
        if os.path.isdir(directory):
            for file in os.listdir(directory):
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(directory, file))
    
    return sorted(csv_files)
