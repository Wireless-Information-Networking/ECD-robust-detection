#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: data_loader.py
Author: Javier del Río
Date: 2025-10-08
Description: 
    Generalized data loading utilities for RFID tag analysis.
    Provides functions to load multiple CSV files and assign labels for machine learning models.
    Designed to work with various AI models including XGBoost, SVM, Random Forest, and Neural Networks.
    Does not include segmentation or feature extraction - focuses only on data loading and labeling.

License: MIT License
Dependencies: numpy, csv_data_loader (local)
"""

import numpy as np
import os
from typing import List, Dict, Tuple, Union

from csv_data_loader import extract_tag_data, get_csv_files_from_directory


def load_file_with_label(file_path: str, label: int) -> Tuple[List[Dict], List[int]]:
    """
    Load a CSV file, extract tag data, and assign labels to all tags from the file.
    
    :param file_path: Path to the CSV file
    :param label: Label to assign to all tags from this file
    :return: Tuple of (tag_data_list, labels_list)
    """
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping...")
        return [], []
    
    try:
        tags_data = extract_tag_data(file_path)
        if not tags_data:
            print(f"Warning: No tag data found in {file_path}")
            return [], []
        
        tag_data_list = list(tags_data.values())
        labels_list = [label] * len(tag_data_list)
        
        print(f"Loaded {len(tag_data_list)} tags from {os.path.basename(file_path)} with label {label}")
        return tag_data_list, labels_list
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return [], []


def load_multiple_files(file_configs: List[Dict[str, Union[str, int]]]) -> Tuple[List[Dict], List[int]]:
    """
    Load multiple CSV files with their corresponding labels.
    
    :param file_configs: List of dictionaries with 'path' and 'label' keys
    :return: Tuple of (all_tag_data, all_labels)
    """
    all_tag_data = []
    all_labels = []
    
    for config in file_configs:
        file_path = config['path']
        label = config['label']
        
        tag_data_list, labels_list = load_file_with_label(file_path, label)
        all_tag_data.extend(tag_data_list)
        all_labels.extend(labels_list)
    
    print(f"Total tags loaded: {len(all_tag_data)}")
    if all_labels:
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        print(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    return all_tag_data, all_labels


def load_directory_with_label(directory_path: str, label: int, 
                              file_pattern: str = "*.csv") -> Tuple[List[Dict], List[int]]:
    """
    Load all CSV files from a directory and assign the same label to all tags.
    
    :param directory_path: Path to directory containing CSV files
    :param label: Label to assign to all tags from this directory
    :param file_pattern: File pattern to match (default: "*.csv")
    :return: Tuple of (all_tag_data, all_labels)
    """
    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} not found. Skipping...")
        return [], []
    
    try:
        csv_files = get_csv_files_from_directory(directory_path)
        if not csv_files:
            print(f"Warning: No CSV files found in {directory_path}")
            return [], []
        
        all_tag_data = []
        all_labels = []
        
        for csv_file in csv_files:
            tag_data_list, labels_list = load_file_with_label(csv_file, label)
            all_tag_data.extend(tag_data_list)
            all_labels.extend(labels_list)
        
        print(f"Loaded {len(all_tag_data)} tags from {len(csv_files)} files in {os.path.basename(directory_path)} with label {label}")
        return all_tag_data, all_labels
        
    except Exception as e:
        print(f"Error loading directory {directory_path}: {e}")
        return [], []


def load_dataset_from_config(dataset_config: Dict) -> Tuple[List[Dict], List[int]]:
    """
    Load dataset from a configuration dictionary.
    
    :param dataset_config: Configuration with 'files' and/or 'directories' keys
    :return: Tuple of (all_tag_data, all_labels)
    
    Example config:
    {
        'files': [
            {'path': 'data/file1.csv', 'label': 0},
            {'path': 'data/file2.csv', 'label': 1}
        ],
        'directories': [
            {'path': 'data/class0/', 'label': 0},
            {'path': 'data/class1/', 'label': 1}
        ]
    }
    """
    all_tag_data = []
    all_labels = []
    
    # Load individual files
    if 'files' in dataset_config:
        files_data, files_labels = load_multiple_files(dataset_config['files'])
        all_tag_data.extend(files_data)
        all_labels.extend(files_labels)
    
    # Load directories
    if 'directories' in dataset_config:
        for dir_config in dataset_config['directories']:
            dir_path = dir_config['path']
            dir_label = dir_config['label']
            dir_data, dir_labels = load_directory_with_label(dir_path, dir_label)
            all_tag_data.extend(dir_data)
            all_labels.extend(dir_labels)
    
    print(f"Total dataset size: {len(all_tag_data)} tags")
    if all_labels:
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        print(f"Final label distribution: {dict(zip(unique_labels, counts))}")
    
    return all_tag_data, all_labels


def get_dataset_info(tag_data_list: List[Dict], labels: List[int]) -> Dict:
    """
    Get information about the loaded dataset.
    
    :param tag_data_list: List of tag data dictionaries
    :param labels: List of labels
    :return: Dictionary with dataset information
    """
    if not tag_data_list:
        return {'num_samples': 0, 'num_tags': 0, 'label_distribution': {}}
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Calculate basic statistics for each tag
    sample_counts = []
    durations = []
    
    for tag_data in tag_data_list:
        if 'timestamp' in tag_data:
            timestamps = np.array(tag_data['timestamp'])
            sample_counts.append(len(timestamps))
            if len(timestamps) > 1:
                durations.append(timestamps[-1] - timestamps[0])
    
    info = {
        'num_tags': len(tag_data_list),
        'num_samples': len(labels),
        'label_distribution': dict(zip(unique_labels, counts)),
        'class_balance': counts.min() / counts.max() if len(counts) > 1 else 1.0,
        'sample_statistics': {
            'avg_samples_per_tag': np.mean(sample_counts) if sample_counts else 0,
            'min_samples_per_tag': np.min(sample_counts) if sample_counts else 0,
            'max_samples_per_tag': np.max(sample_counts) if sample_counts else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'min_duration': np.min(durations) if durations else 0,
            'max_duration': np.max(durations) if durations else 0
        }
    }
    
    return info


def print_dataset_summary(tag_data_list: List[Dict], labels: List[int]):
    """
    Print a summary of the loaded dataset.
    
    :param tag_data_list: List of tag data dictionaries
    :param labels: List of labels
    """
    info = get_dataset_info(tag_data_list, labels)
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total tags: {info['num_tags']}")
    print(f"Label distribution: {info['label_distribution']}")
    print(f"Class balance ratio: {info['class_balance']:.3f}")
    
    if info['sample_statistics']['avg_samples_per_tag'] > 0:
        stats = info['sample_statistics']
        print(f"\nSample statistics per tag:")
        print(f"  Average samples: {stats['avg_samples_per_tag']:.1f}")
        print(f"  Sample range: {stats['min_samples_per_tag']} - {stats['max_samples_per_tag']}")
        print(f"  Average duration: {stats['avg_duration']:.3f}s")
        print(f"  Duration range: {stats['min_duration']:.3f}s - {stats['max_duration']:.3f}s")
    
    print("="*50)


def validate_dataset(tag_data_list: List[Dict], labels: List[int]) -> bool:
    """
    Validate that the dataset is properly formatted and consistent.
    
    :param tag_data_list: List of tag data dictionaries
    :param labels: List of labels
    :return: True if dataset is valid, False otherwise
    """
    if not tag_data_list:
        print("Error: No tag data provided")
        return False
    
    if len(tag_data_list) != len(labels):
        print(f"Error: Mismatch between tag data ({len(tag_data_list)}) and labels ({len(labels)})")
        return False
    
    # Check that all tag data has required fields
    required_fields = ['timestamp', 'rssi', 'phase']
    for i, tag_data in enumerate(tag_data_list):
        for field in required_fields:
            if field not in tag_data:
                print(f"Error: Tag {i} missing required field '{field}'")
                return False
            
            if len(tag_data[field]) == 0:
                print(f"Error: Tag {i} has empty '{field}' data")
                return False
    
    print("Dataset validation: PASSED")
    return True


# Example usage and testing
if __name__ == "__main__":
    print("=== DATA LOADER EXAMPLE ===")
    
    # Example 1: Load individual files
    print("\n1. Loading individual files...")
    file_configs = [
        {'path': 'data/dynamic.csv', 'label': 1},
        {'path': 'data/static.csv', 'label': 0}
    ]
    
    tag_data, labels = load_multiple_files(file_configs)
    
    if tag_data:
        # Validate and summarize
        if validate_dataset(tag_data, labels):
            print_dataset_summary(tag_data, labels)
    
    # Example 2: Load from configuration
    print("\n2. Loading from configuration...")
    config = {
        'files': [
            {'path': 'data/dynamic.csv', 'label': 1},
            {'path': 'data/static.csv', 'label': 0}
        ],
        'directories': [
            {'path': 'data/test/', 'label': 1}
        ]
    }
    
    tag_data_config, labels_config = load_dataset_from_config(config)
    
    if tag_data_config:
        print_dataset_summary(tag_data_config, labels_config)
    
    # Example 3: Load directory
    print("\n3. Loading from directory...")
    dir_data, dir_labels = load_directory_with_label('data/test/', label=1)
    
    if dir_data:
        print_dataset_summary(dir_data, dir_labels)