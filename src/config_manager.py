#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: config_manager.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Configuration management utilities for RFID processing parameters and expected results.
    Provides functions to load/save optimization parameters, expected arc counts, and 
    processing configurations with robust file handling and validation. Supports both
    segmentation parameters and expected detection results for automated optimization
    workflows and parameter tuning processes.

License: MIT License
"""

import json
import os
import numpy as np
from typing import Dict, Any, List, Optional

def load_expected_arcs_config(config_path: str = 'data/test/file_arc_num.json') -> Dict[str, Any]:
    """
    Loads the expected arcs configuration from JSON file.
    
    :param config_path: Path to the configuration JSON file
    :return: Dictionary with expected arc counts per file and tag
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration if file doesn't exist
        print(f"Configuration file {config_path} not found. Using default configuration.")
        return {
            'dynamic.csv': {
                'default': 10
            }
        }

def save_expected_arcs_config(config_data: Dict[str, Any], config_path: str = 'data/test/file_arc_num.json'):
    """
    Saves the expected arcs configuration to JSON file.
    
    :param config_data: Dictionary with expected arc counts per file and tag
    :param config_path: Path to save the configuration JSON file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")

def get_expected_arcs(csv_file: str, tag_id: str, expected_config: Dict[str, Any]) -> int:
    """
    Gets the expected number of arcs for a specific file and tag.
    
    :param csv_file: Path to CSV file
    :param tag_id: Tag identifier
    :param expected_config: Configuration dictionary
    :return: Expected number of arcs
    """
    file_name = os.path.basename(csv_file)
    
    if file_name in expected_config:
        file_config = expected_config[file_name]
        if tag_id in file_config:
            return file_config[tag_id]
        elif 'default' in file_config:
            return file_config['default']
    
    # If no specific configuration exists, use default value
    return 5

def save_optimized_parameters(best_params: List[float], output_dir: str = 'output_data') -> str:
    """
    Saves optimized parameters to a JSON file.
    
    :param best_params: Array of optimized parameters
    :param output_dir: Directory to save the file
    :return: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    params_dict = {
        'abs_threshold': float(best_params[0]),
        'stat_threshold': float(best_params[1]),
        'num_interp_points': int(best_params[2]),
        'smoothing_sigma': float(best_params[3]),
        'optimization_date': str(np.datetime64('now'))
    }
    
    output_path = os.path.join(output_dir, 'optimized_parameters.json')
    with open(output_path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    print(f"Optimized parameters saved to: {output_path}")
    return output_path

def load_optimized_parameters(input_path: str = 'output_data/optimized_parameters.json') -> Optional[Dict[str, Any]]:
    """
    Loads optimized parameters from JSON file.
    
    :param input_path: Path to the optimized parameters file
    :return: Dictionary with optimized parameters or None if file doesn't exist
    """
    if os.path.exists(input_path):
        with open(input_path, 'r') as f:
            params = json.load(f)
        
        print(f"Loading optimized parameters from: {input_path}")
        print("Optimized parameters loaded:")
        for key, value in params.items():
            if key != 'optimization_date':
                print(f"  {key}: {value}")
        
        return params
    else:
        print(f"Optimized parameters file not found: {input_path}")
        print("Using default parameters...")
        return {
            'abs_threshold': 1.0,
            'stat_threshold': 8.0,
            'num_interp_points': 200,
            'smoothing_sigma': 4.0
        }
