#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: optimize_params.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Parameter optimization script for RFID tag segmentation and arc detection.
    Uses grid search and differential evolution algorithms to find optimal parameters
    for segmentation thresholds, interpolation points, and smoothing factors based
    on expected arc counts from configuration files.

License: MIT License
Dependencies: numpy, matplotlib, scipy, json, timediff, interpolation, file_operations
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import differential_evolution
from typing import Dict, Any
# Local imports from existing modules
from config_manager import load_expected_arcs_config, get_expected_arcs, save_optimized_parameters
from csv_data_loader import extract_tag_data, get_csv_files_from_directory
from arc_segmentation import segment_tag_data_into_arcs, analyze_arc_statistics, print_arc_statistics, plot_arc_segmentation_results

def process_files_to_arcs(data_folder: str, abs_threshold: float = 1.0, stat_threshold: float = 2.0, 
                         num_interp_points: int = 200, smoothing_sigma: float = 2.0,
                         min_arc_duration: float = 0.1, min_arc_samples: int = 5,
                         minima_min_distance: int = 10, minima_prominence: float = 0.1) -> Dict[str, Any]:
    """
    Processes all CSV files in a folder and converts them directly to arcs using arc_segmentation.
    
    :param data_folder: Folder containing CSV files
    :param abs_threshold: Absolute time difference threshold for segmentation
    :param stat_threshold: Statistical threshold multiplier for segmentation
    :param num_interp_points: Number of interpolation points
    :param smoothing_sigma: Gaussian smoothing parameter
    :param min_arc_duration: Minimum duration for a valid arc
    :param min_arc_samples: Minimum number of samples for a valid arc
    :param minima_min_distance: Minimum distance between local minima
    :param minima_prominence: Prominence threshold for minima detection
    :return: Dictionary with arc data for all files
    """
    # Get all CSV files from directory
    csv_files = get_csv_files_from_directory(data_folder)
    print(f"Found {len(csv_files)} CSV files in {data_folder}")
    
    all_file_arcs = {}
    
    for csv_file in csv_files:
        print(f"\nProcessing: {os.path.basename(csv_file)}")
        
        try:
            # Extract tag data using csv_data_loader
            tag_data_dict = extract_tag_data(csv_file)
            
            if not tag_data_dict:
                print(f"  Could not extract data from {csv_file}")
                continue
            
            file_arcs = {}
            
            # Process each tag in the file
            for tag_id, tag_data in tag_data_dict.items():
                #print(f"  Processing tag {tag_id}: {len(tag_data['timestamp'])} samples")
                
                # Convert to numpy arrays if needed
                for key in ['timestamp', 'rssi', 'phase']:
                    if key in tag_data:
                        tag_data[key] = np.array(tag_data[key])
                
                # Use arc_segmentation function directly
                arcs = segment_tag_data_into_arcs(
                    tag_data,
                    abs_threshold=abs_threshold,
                    stat_threshold=stat_threshold,
                    num_interp_points=num_interp_points,
                    smoothing_sigma=smoothing_sigma,
                    min_arc_duration=min_arc_duration,
                    min_arc_samples=min_arc_samples,
                    minima_min_distance=minima_min_distance,
                    minima_prominence=minima_prominence,
                    verbose=False  # Suppress verbose output for optimization
                )
                
                file_arcs[tag_id] = {
                    'original_data': tag_data,
                    'arcs': arcs,
                    'num_arcs': len(arcs)
                }
                
                #print(f"    -> {len(arcs)} arcs detected")
            
            all_file_arcs[csv_file] = file_arcs
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
    
    return all_file_arcs

def objective_function(params, data_folder, expected_config):
    """
    Objective function that calculates error between expected and found arcs.
    
    :param params: Parameter array [abs_threshold, stat_threshold, num_interp_points, smoothing_sigma]
    :param data_folder: Folder containing test data
    :param expected_config: Expected arc configuration
    :return: Average error across all files and tags
    """
    abs_threshold, stat_threshold, num_interp_points, smoothing_sigma = params
    
    # Validate parameters
    if abs_threshold <= 0 or stat_threshold <= 0 or num_interp_points < 50 or smoothing_sigma <= 0:
        return 1000  # Penalty for invalid parameters
    
    total_error = 0
    total_comparisons = 0
    
    try:
        # Process files to arcs with new parameters
        all_arcs = process_files_to_arcs(
            data_folder,
            abs_threshold=abs_threshold,
            stat_threshold=stat_threshold,
            num_interp_points=int(num_interp_points),
            smoothing_sigma=smoothing_sigma
        )
        
        # Calculate error for each file and tag
        for csv_file, file_data in all_arcs.items():
            for tag_id, tag_data in file_data.items():
                expected_arcs = get_expected_arcs(csv_file, tag_id, expected_config)
                found_arcs = len(tag_data['arcs'])
                
                # Absolute error
                error = abs(expected_arcs - found_arcs)
                total_error += error
                total_comparisons += 1
        
        # Return average error
        return total_error / max(total_comparisons, 1)
    
    except Exception as e:
        print(f"Error in objective function evaluation: {e}")
        return 1000  # Penalty for error

def mre_function(params, data_folder, expected_config):
    """
    Mean Relative Error function that calculates error between expected and found arcs.
    
    :param params: Parameter array [abs_threshold, stat_threshold, num_interp_points, smoothing_sigma]
    :param data_folder: Folder containing test data
    :param expected_config: Expected arc configuration
    :return: Average absolute error across all files and tags (Mean Relative Error)
    """
    abs_threshold, stat_threshold, num_interp_points, smoothing_sigma = params
    
    # Validate parameters
    if abs_threshold <= 0 or stat_threshold <= 0 or num_interp_points < 50 or smoothing_sigma <= 0:
        return 1000  # Penalty for invalid parameters
    
    total_error = 0
    total_comparisons = 0
    
    try:
        # Process files to arcs with new parameters
        all_arcs = process_files_to_arcs(
            data_folder,
            abs_threshold=abs_threshold,
            stat_threshold=stat_threshold,
            num_interp_points=int(num_interp_points),
            smoothing_sigma=smoothing_sigma
        )
        
        # Calculate error for each file and tag
        for csv_file, file_data in all_arcs.items():
            for tag_id, tag_data in file_data.items():
                expected_arcs = get_expected_arcs(csv_file, tag_id, expected_config)
                found_arcs = len(tag_data['arcs'])
                
                # Absolute error (Mean Relative Error approach)
                error = abs(expected_arcs - found_arcs)
                total_error += error
                total_comparisons += 1
        
        # Return average absolute error
        return total_error / max(total_comparisons, 1)
    
    except Exception as e:
        print(f"Error in MRE function evaluation: {e}")
        return 1000  # Penalty for error

def mse_function(params, data_folder, expected_config):
    """
    Mean Squared Error function that calculates error between expected and found arcs.
    
    :param params: Parameter array [abs_threshold, stat_threshold, num_interp_points, smoothing_sigma]
    :param data_folder: Folder containing test data
    :param expected_config: Expected arc configuration
    :return: Mean squared error across all files and tags
    """
    abs_threshold, stat_threshold, num_interp_points, smoothing_sigma = params
    
    # Validate parameters
    if abs_threshold <= 0 or stat_threshold <= 0 or num_interp_points < 50 or smoothing_sigma <= 0:
        return 1000  # Penalty for invalid parameters
    
    total_squared_error = 0
    total_comparisons = 0
    
    try:
        # Process files to arcs with new parameters
        all_arcs = process_files_to_arcs(
            data_folder,
            abs_threshold=abs_threshold,
            stat_threshold=stat_threshold,
            num_interp_points=int(num_interp_points),
            smoothing_sigma=smoothing_sigma
        )
        
        # Calculate squared error for each file and tag
        for csv_file, file_data in all_arcs.items():
            for tag_id, tag_data in file_data.items():
                expected_arcs = get_expected_arcs(csv_file, tag_id, expected_config)
                found_arcs = len(tag_data['arcs'])
                
                # Squared error (Mean Squared Error approach)
                error = (expected_arcs - found_arcs) ** 2
                total_squared_error += error
                total_comparisons += 1
        
        # Return mean squared error
        return total_squared_error / max(total_comparisons, 1)
    
    except Exception as e:
        print(f"Error in MSE function evaluation: {e}")
        return 1000  # Penalty for error

def optimize_parameters_grid_search(data_folder, expected_config, error_function='mre', verbose=True):
    """
    Optimization using grid search (faster but less precise).
    
    :param data_folder: Folder containing test data
    :param expected_config: Expected arc configuration
    :param error_function: 'mre' for Mean Relative Error or 'mse' for Mean Squared Error
    :param verbose: Whether to print progress
    :return: Tuple of (best_params, best_error)
    """
    print(f"=== GRID SEARCH OPTIMIZATION ({error_function.upper()}) ===")
    
    # Select error function
    if error_function == 'mre':
        objective_func = mre_function
        print("Using Mean Relative Error (MRE) - penalizes all errors equally")
    elif error_function == 'mse':
        objective_func = mse_function
        print("Using Mean Squared Error (MSE) - penalizes larger errors more heavily")
    else:
        raise ValueError("error_function must be 'mre' or 'mse'")
    
    # Define search ranges based on optimized parameters from JSON
    abs_thresholds = np.linspace(1.0, 3.0, 5)
    stat_thresholds = np.linspace(3.0, 8.0, 5)
    interp_points = np.linspace(50, 150, 5, dtype=int)
    smoothing_sigmas = np.linspace(1.0, 3.0, 5)
    
    best_error = float('inf')
    best_params = None
    total_combinations = len(abs_thresholds) * len(stat_thresholds) * len(interp_points) * len(smoothing_sigmas)
    
    print(f"Evaluating {total_combinations} combinations...")
    
    current_combination = 0
    
    for abs_thresh in abs_thresholds:
        for stat_thresh in stat_thresholds:
            for interp_pts in interp_points:
                for smooth_sigma in smoothing_sigmas:
                    current_combination += 1
                    
                    params = [abs_thresh, stat_thresh, interp_pts, smooth_sigma]
                    error = objective_func(params, data_folder, expected_config)
                    
                    if error < best_error:
                        best_error = error
                        best_params = params
                        if verbose:
                            print(f"  New best ({error_function.upper()}): error={error:.3f}, params={params}")
                    
                    if verbose and current_combination % 5 == 0:
                        print(f"  Progress: {current_combination}/{total_combinations} ({100*current_combination/total_combinations:.1f}%)")
    
    return best_params, best_error

def optimize_parameters_differential_evolution(data_folder, expected_config, error_function='mre'):
    """
    Optimization using differential evolution (more precise but slower).
    
    :param data_folder: Folder containing test data
    :param expected_config: Expected arc configuration
    :param error_function: 'mre' for Mean Relative Error or 'mse' for Mean Squared Error
    :return: Tuple of (best_params, best_error)
    """
    print(f"=== DIFFERENTIAL EVOLUTION OPTIMIZATION ({error_function.upper()}) ===")
    
    # Select error function
    if error_function == 'mre':
        objective_func = mre_function
        print("Using Mean Relative Error (MRE) - treats all errors equally")
    elif error_function == 'mse':
        objective_func = mse_function
        print("Using Mean Squared Error (MSE) - penalizes larger errors quadratically")
    else:
        raise ValueError("error_function must be 'mre' or 'mse'")
    
    # Define parameter bounds based on reasonable ranges
    bounds = [
        (0.5, 3.0),    # abs_threshold
        (1.0, 10.0),   # stat_threshold  
        (50, 300),     # num_interp_points
        (0.5, 5.0)     # smoothing_sigma
    ]
    
    def objective_wrapper(params):
        return objective_func(params, data_folder, expected_config)
    
    print("Running optimization...")
    result = differential_evolution(
        objective_wrapper,
        bounds,
        maxiter=30,
        popsize=8,
        seed=42,
        disp=True
    )
    
    return result.x, result.fun

def evaluate_parameters(params, data_folder, expected_config, show_both_errors=True):
    """
    Evaluates a parameter set and shows detailed results with both MRE and MSE.
    
    :param params: Parameter array to evaluate
    :param data_folder: Folder containing test data
    :param expected_config: Expected arc configuration
    :param show_both_errors: Whether to show both MRE and MSE metrics
    :return: Tuple of (arc_results, mre_error, mse_error)
    """
    abs_threshold, stat_threshold, num_interp_points, smoothing_sigma = params
    
    print(f"\n=== PARAMETER EVALUATION ===")
    print(f"abs_threshold: {abs_threshold:.3f}")
    print(f"stat_threshold: {stat_threshold:.3f}")
    print(f"num_interp_points: {int(num_interp_points)}")
    print(f"smoothing_sigma: {smoothing_sigma:.3f}")
    
    # Process with these parameters
    all_arcs = process_files_to_arcs(
        data_folder,
        abs_threshold=abs_threshold,
        stat_threshold=stat_threshold,
        num_interp_points=int(num_interp_points),
        smoothing_sigma=smoothing_sigma
    )
    
    # Show detailed comparison
    print(f"\n{'File':<20} {'Tag':<25} {'Expected':<10} {'Found':<12} {'Abs Error':<10} {'Sq Error':<10}")
    print("-" * 100)
    
    total_abs_error = 0
    total_squared_error = 0
    total_comparisons = 0
    
    for csv_file, file_data in all_arcs.items():
        file_name = os.path.basename(csv_file)
        for tag_id, tag_data in file_data.items():
            expected = get_expected_arcs(csv_file, tag_id, expected_config)
            found = len(tag_data['arcs'])
            abs_error = abs(expected - found)
            sq_error = (expected - found) ** 2
            
            print(f"{file_name:<20} {tag_id[:25]:<25} {expected:<10} {found:<12} {abs_error:<10} {sq_error:<10.2f}")
            
            total_abs_error += abs_error
            total_squared_error += sq_error
            total_comparisons += 1
    
    mre = total_abs_error / max(total_comparisons, 1)
    mse = total_squared_error / max(total_comparisons, 1)
    rmse = np.sqrt(mse)
    
    print(f"\nError Metrics:")
    print(f"  Mean Relative Error (MRE): {mre:.3f}")
    print(f"  Mean Squared Error (MSE): {mse:.3f}")
    print(f"  Root Mean Square Error (RMSE): {rmse:.3f}")
    
    if show_both_errors:
        print(f"\nError Function Comparison:")
        print(f"  MRE treats all errors equally: |expected - found|")
        print(f"  MSE penalizes larger errors: (expected - found)²")
        print(f"  RMSE is MSE in original scale: √MSE")
    
    return all_arcs, mre, mse

def mre_function_extended(params, data_folder, expected_config):
    """
    Extended Mean Relative Error function with 6 parameters.
    
    :param params: Parameter array [abs_threshold, stat_threshold, num_interp_points, smoothing_sigma, min_arc_duration, min_arc_samples]
    :param data_folder: Folder containing test data
    :param expected_config: Expected arc configuration
    :return: Average absolute error across all files and tags
    """
    abs_threshold, stat_threshold, num_interp_points, smoothing_sigma, min_arc_duration, min_arc_samples = params
    
    # Validate parameters
    if (abs_threshold <= 0 or stat_threshold <= 0 or num_interp_points < 50 or 
        smoothing_sigma <= 0 or min_arc_duration <= 0 or min_arc_samples < 1):
        return 1000  # Penalty for invalid parameters
    
    total_error = 0
    total_comparisons = 0
    
    try:
        # Process files to arcs with new parameters (including extended ones)
        all_arcs = process_files_to_arcs_extended(
            data_folder,
            abs_threshold=abs_threshold,
            stat_threshold=stat_threshold,
            num_interp_points=int(num_interp_points),
            smoothing_sigma=smoothing_sigma,
            min_arc_duration=min_arc_duration,
            min_arc_samples=int(min_arc_samples)
        )
        
        # Calculate error for each file and tag
        for csv_file, file_data in all_arcs.items():
            for tag_id, tag_data in file_data.items():
                expected_arcs = get_expected_arcs(csv_file, tag_id, expected_config)
                found_arcs = len(tag_data['arcs'])
                
                # Absolute error
                error = abs(expected_arcs - found_arcs)
                total_error += error
                total_comparisons += 1
        
        # Return average error
        return total_error / max(total_comparisons, 1)
    
    except Exception as e:
        print(f"Error in extended MRE function evaluation: {e}")
        return 1000  # Penalty for error

def mse_function_extended(params, data_folder, expected_config):
    """
    Extended Mean Squared Error function with 6 parameters.
    
    :param params: Parameter array [abs_threshold, stat_threshold, num_interp_points, smoothing_sigma, min_arc_duration, min_arc_samples]
    :param data_folder: Folder containing test data
    :param expected_config: Expected arc configuration
    :return: Mean squared error across all files and tags
    """
    abs_threshold, stat_threshold, num_interp_points, smoothing_sigma, min_arc_duration, min_arc_samples = params
    
    # Validate parameters
    if (abs_threshold <= 0 or stat_threshold <= 0 or num_interp_points < 50 or 
        smoothing_sigma <= 0 or min_arc_duration <= 0 or min_arc_samples < 1):
        return 1000  # Penalty for invalid parameters
    
    total_squared_error = 0
    total_comparisons = 0
    
    try:
        # Process files to arcs with new parameters (including extended ones)
        all_arcs = process_files_to_arcs_extended(
            data_folder,
            abs_threshold=abs_threshold,
            stat_threshold=stat_threshold,
            num_interp_points=int(num_interp_points),
            smoothing_sigma=smoothing_sigma,
            min_arc_duration=min_arc_duration,
            min_arc_samples=int(min_arc_samples)
        )
        
        # Calculate squared error for each file and tag
        for csv_file, file_data in all_arcs.items():
            for tag_id, tag_data in file_data.items():
                expected_arcs = get_expected_arcs(csv_file, tag_id, expected_config)
                found_arcs = len(tag_data['arcs'])
                
                # Squared error
                error = (expected_arcs - found_arcs) ** 2
                total_squared_error += error
                total_comparisons += 1
        
        # Return mean squared error
        return total_squared_error / max(total_comparisons, 1)
    
    except Exception as e:
        print(f"Error in extended MSE function evaluation: {e}")
        return 1000  # Penalty for error

def optimize_parameters_grid_search_extended(data_folder, expected_config, error_function='mre', verbose=True):
    """
    Extended grid search optimization with 6 parameters.
    
    :param data_folder: Folder containing test data
    :param expected_config: Expected arc configuration
    :param error_function: 'mre' for Mean Relative Error or 'mse' for Mean Squared Error
    :param verbose: Whether to print progress
    :return: Tuple of (best_params, best_error)
    """
    print(f"=== EXTENDED GRID SEARCH OPTIMIZATION (6 PARAMETERS, {error_function.upper()}) ===")
    
    # Select error function
    if error_function == 'mre':
        objective_func = mre_function_extended
        print("Using Mean Relative Error (MRE) - treats all errors equally")
    elif error_function == 'mse':
        objective_func = mse_function_extended
        print("Using Mean Squared Error (MSE) - penalizes larger errors more heavily")
    else:
        raise ValueError("error_function must be 'mre' or 'mse'")
    
    # Define search ranges for all 6 parameters
    abs_thresholds = np.linspace(1.0, 3.0, 3)  # Reduced grid size due to increased dimensionality
    stat_thresholds = np.linspace(3.0, 8.0, 3)
    interp_points = np.linspace(50, 150, 3, dtype=int)
    smoothing_sigmas = np.linspace(1.0, 3.0, 3)
    min_arc_durations = np.linspace(0.05, 0.3, 3)
    min_arc_samples_list = np.array([3, 5, 10], dtype=int)
    
    best_error = float('inf')
    best_params = None
    total_combinations = (len(abs_thresholds) * len(stat_thresholds) * len(interp_points) * 
                         len(smoothing_sigmas) * len(min_arc_durations) * len(min_arc_samples_list))
    
    print(f"Evaluating {total_combinations} combinations...")
    
    current_combination = 0
    
    for abs_thresh in abs_thresholds:
        for stat_thresh in stat_thresholds:
            for interp_pts in interp_points:
                for smooth_sigma in smoothing_sigmas:
                    for min_duration in min_arc_durations:
                        for min_samples in min_arc_samples_list:
                            current_combination += 1
                            
                            params = [abs_thresh, stat_thresh, interp_pts, smooth_sigma, min_duration, min_samples]
                            error = objective_func(params, data_folder, expected_config)
                            
                            if error < best_error:
                                best_error = error
                                best_params = params
                                if verbose:
                                    print(f"  New best ({error_function.upper()}): error={error:.3f}")
                                    print(f"    abs_threshold: {params[0]:.3f}")
                                    print(f"    stat_threshold: {params[1]:.3f}")
                                    print(f"    num_interp_points: {int(params[2])}")
                                    print(f"    smoothing_sigma: {params[3]:.3f}")
                                    print(f"    min_arc_duration: {params[4]:.3f}")
                                    print(f"    min_arc_samples: {int(params[5])}")
                            
                            if verbose and current_combination % 10 == 0:
                                print(f"  Progress: {current_combination}/{total_combinations} ({100*current_combination/total_combinations:.1f}%)")
    
    return best_params, best_error

def optimize_parameters_differential_evolution_extended(data_folder, expected_config, error_function='mre'):
    """
    Extended differential evolution optimization with 6 parameters.
    
    :param data_folder: Folder containing test data
    :param expected_config: Expected arc configuration
    :param error_function: 'mre' for Mean Relative Error or 'mse' for Mean Squared Error
    :return: Tuple of (best_params, best_error)
    """
    print(f"=== EXTENDED DIFFERENTIAL EVOLUTION OPTIMIZATION (6 PARAMETERS, {error_function.upper()}) ===")
    
    # Select error function
    if error_function == 'mre':
        objective_func = mre_function_extended
        print("Using Mean Relative Error (MRE) - treats all errors equally")
    elif error_function == 'mse':
        objective_func = mse_function_extended
        print("Using Mean Squared Error (MSE) - penalizes larger errors quadratically")
    else:
        raise ValueError("error_function must be 'mre' or 'mse'")
    
    # Define parameter bounds for all 6 parameters
    bounds = [
        (0.5, 3.0),     # abs_threshold
        (1.0, 10.0),    # stat_threshold  
        (50, 300),      # num_interp_points
        (0.5, 5.0),     # smoothing_sigma
        (0.05, 0.5),    # min_arc_duration
        (2, 15)         # min_arc_samples
    ]
    
    def objective_wrapper(params):
        return objective_func(params, data_folder, expected_config)
    
    print("Running extended optimization...")
    print("Parameter bounds:")
    param_names = ['abs_threshold', 'stat_threshold', 'num_interp_points', 
                   'smoothing_sigma', 'min_arc_duration', 'min_arc_samples']
    for i, (name, bound) in enumerate(zip(param_names, bounds)):
        print(f"  {name}: {bound[0]} to {bound[1]}")
    
    result = differential_evolution(
        objective_wrapper,
        bounds,
        maxiter=40,  # Increased iterations for more parameters
        popsize=12,  # Increased population size
        seed=42,
        disp=True
    )
    
    return result.x, result.fun

def evaluate_parameters_extended(params, data_folder, expected_config):
    """
    Extended parameter evaluation with 6 parameters.
    
    :param params: Parameter array with 6 elements
    :param data_folder: Folder containing test data
    :param expected_config: Expected arc configuration
    :return: Tuple of (arc_results, rmse)
    """
    abs_threshold, stat_threshold, num_interp_points, smoothing_sigma, min_arc_duration, min_arc_samples = params
    
    print(f"\n=== EXTENDED PARAMETER EVALUATION ===")
    print(f"abs_threshold: {abs_threshold:.3f}")
    print(f"stat_threshold: {stat_threshold:.3f}")
    print(f"num_interp_points: {int(num_interp_points)}")
    print(f"smoothing_sigma: {smoothing_sigma:.3f}")
    print(f"min_arc_duration: {min_arc_duration:.3f}")
    print(f"min_arc_samples: {int(min_arc_samples)}")
    
    # Process with these parameters
    all_arcs = process_files_to_arcs_extended(
        data_folder,
        abs_threshold=abs_threshold,
        stat_threshold=stat_threshold,
        num_interp_points=int(num_interp_points),
        smoothing_sigma=smoothing_sigma,
        min_arc_duration=min_arc_duration,
        min_arc_samples=int(min_arc_samples)
    )
    
    # Show detailed comparison
    print(f"\n{'File':<20} {'Tag':<25} {'Expected':<10} {'Found':<12} {'Error':<8}")
    print("-" * 80)
    
    total_error = 0
    total_comparisons = 0
    
    for csv_file, file_data in all_arcs.items():
        file_name = os.path.basename(csv_file)
        for tag_id, tag_data in file_data.items():
            expected = get_expected_arcs(csv_file, tag_id, expected_config)
            found = len(tag_data['arcs'])
            error = abs(expected - found)
            
            print(f"{file_name:<20} {tag_id[:25]:<25} {expected:<10} {found:<12} {error:<8}")
            
            total_error += error ** 2
            total_comparisons += 1
    
    rmse = np.sqrt(total_error / max(total_comparisons, 1))
    print(f"\nRoot Mean Square Error (RMSE): {rmse:.3f}")
    
    return all_arcs, rmse

def plot_optimization_results(all_arcs, output_dir='output_plots'):
    """
    Creates plots for optimization results using arc_segmentation functions.
    
    :param all_arcs: Dictionary with arc data from all files
    :param output_dir: Directory to save plots
    """
    print("\nGenerating optimization result plots...")
    
    # Create summary statistics for all arcs
    all_arc_list = []
    for csv_file, file_data in all_arcs.items():
        for tag_id, tag_data in file_data.items():
            all_arc_list.extend(tag_data['arcs'])
    
    if all_arc_list:
        # Use arc_segmentation functions for analysis and plotting
        stats = analyze_arc_statistics(all_arc_list)
        print_arc_statistics(stats)
        
        # Plot individual files
        for csv_file, file_data in all_arcs.items():
            file_name = os.path.basename(csv_file).replace('.csv', '')
            print(f"Creating plot for: {file_name}")
            
            for tag_id, tag_data in file_data.items():
                if tag_data['arcs']:
                    plot_arc_segmentation_results(
                        tag_data['original_data'],
                        tag_data['arcs'],
                        f"{file_name}_{tag_id}",
                        save_plot=True,
                        output_dir=output_dir
                    )

def adaptive_parameter_optimization(data_folder='data/test', optimization_method='grid', error_function='mre'):
    """
    Main function for adaptive parameter optimization.
    
    :param data_folder: Folder containing test data
    :param optimization_method: 'grid' or 'differential_evolution'
    :param error_function: 'mre' for Mean Relative Error or 'mse' for Mean Squared Error
    :return: Tuple of (best_params, final_results)
    """
    print(f"=== ADAPTIVE PARAMETER OPTIMIZATION ({error_function.upper()}) ===\n")
    
    # Load expected configuration using config_manager
    expected_config = load_expected_arcs_config()
    print("Expected arc configuration:")
    for file_name, file_config in expected_config.items():
        print(f"  {file_name}: {file_config}")
    
    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
        return None, None
    
    # Optimize parameters
    if optimization_method == 'grid':
        best_params, best_error = optimize_parameters_grid_search(data_folder, expected_config, error_function)
    elif optimization_method == 'differential_evolution':
        best_params, best_error = optimize_parameters_differential_evolution(data_folder, expected_config, error_function)
    else:
        raise ValueError("optimization_method must be 'grid' or 'differential_evolution'")
    
    print(f"\n=== BEST PARAMETERS FOUND ({error_function.upper()}) ===")
    print(f"abs_threshold: {best_params[0]:.3f}")
    print(f"stat_threshold: {best_params[1]:.3f}")
    print(f"num_interp_points: {int(best_params[2])}")
    print(f"smoothing_sigma: {best_params[3]:.3f}")
    print(f"Final {error_function.upper()} error: {best_error:.3f}")
    
    # Evaluate with best parameters (show both error types)
    final_results, mre_error, mse_error = evaluate_parameters(best_params, data_folder, expected_config, show_both_errors=True)
    
    return best_params, final_results

def main_optimized(error_function='mre'):
    """
    Main function with parameter optimization workflow.
    
    :param error_function: 'mre' for Mean Relative Error or 'mse' for Mean Squared Error
    :return: Tuple of (best_params, optimized_results)
    """
    print(f"=== OPTIMIZED SEGMENTATION INTO PASSES AND ARCS ({error_function.upper()}) ===\n")
    
    # Configure data folder
    data_folder = "data/test"
    
    # Run optimization
    best_params, optimized_results = adaptive_parameter_optimization(
        data_folder=data_folder,
        optimization_method='differential_evolution',  # Change to 'grid' for faster but less precise optimization
        error_function=error_function
    )
    
    if best_params is None:
        return None, None
    
    # Save optimized parameters using config_manager
    save_optimized_parameters(best_params)
    
    # Generate visualizations with optimized parameters
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS WITH OPTIMIZED PARAMETERS...")
    
    plot_optimization_results(optimized_results)
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETED!")
    print(f"Best parameters ({error_function.upper()}):")
    print(f"  abs_threshold: {best_params[0]:.3f}")
    print(f"  stat_threshold: {best_params[1]:.3f}")
    print(f"  num_interp_points: {int(best_params[2])}")
    print(f"  smoothing_sigma: {best_params[3]:.3f}")
    
    return best_params, optimized_results

def main_extended_optimization(error_function='mre'):
    """
    Main function for extended parameter optimization with 6 parameters.
    
    :param error_function: 'mre' for Mean Relative Error or 'mse' for Mean Squared Error
    :return: Tuple of (best_params, optimized_results)
    """
    print(f"=== EXTENDED OPTIMIZATION: 6 PARAMETERS ({error_function.upper()}) ===\n")
    print("Optimizing parameters:")
    print("  1. abs_threshold")
    print("  2. stat_threshold") 
    print("  3. num_interp_points")
    print("  4. smoothing_sigma")
    print("  5. min_arc_duration")
    print("  6. min_arc_samples")
    print(f"\nUsing {error_function.upper()} as optimization criterion")
    print()
    
    # Configure data folder
    data_folder = "data/test"
    
    # Run extended optimization
    best_params, optimized_results = adaptive_parameter_optimization_extended(
        data_folder=data_folder,
        optimization_method='differential_evolution',  # Recommended for higher dimensionality
        error_function=error_function
    )
    
    if best_params is None:
        print("Extended optimization failed.")
        return None, None
    
    # Save extended optimized parameters
    save_extended_optimized_parameters(best_params)
    
    # Generate visualizations with optimized parameters
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS WITH EXTENDED OPTIMIZED PARAMETERS...")
    
    plot_optimization_results(optimized_results, output_dir='output_plots_extended')
    
    print("\n" + "="*60)
    print("EXTENDED OPTIMIZATION COMPLETED!")
    print(f"Best parameters found ({error_function.upper()}):")
    print(f"  abs_threshold: {best_params[0]:.3f}")
    print(f"  stat_threshold: {best_params[1]:.3f}")
    print(f"  num_interp_points: {int(best_params[2])}")
    print(f"  smoothing_sigma: {best_params[3]:.3f}")
    print(f"  min_arc_duration: {best_params[4]:.3f}")
    print(f"  min_arc_samples: {int(best_params[5])}")
    
    print(f"\nParameters saved to: output_data/extended_optimized_parameters.json")
    print(f"Plots saved to: output_plots_extended/")
    
    return best_params, optimized_results

# Modified main section to include both optimization options and error functions
if __name__ == "__main__":
    import sys
    
    print("=== RFID PARAMETER OPTIMIZATION SUITE ===\n")
    print("Available optimization modes:")
    print("  1. Standard optimization (4 parameters)")
    print("  2. Extended optimization (6 parameters)")
    print("\nAvailable error functions:")
    print("  - MRE (Mean Relative Error): treats all errors equally")
    print("  - MSE (Mean Squared Error): penalizes larger errors more heavily")
    
    # Parse command line arguments
    extended_mode = False
    error_function = 'mre'  # Default
    
    for arg in sys.argv[1:]:
        if arg == 'extended':
            extended_mode = True
        elif arg in ['mre', 'mse']:
            error_function = arg
    
    if extended_mode:
        print(f"\nRunning EXTENDED optimization with 6 parameters using {error_function.upper()}...")
        best_params, results = main_extended_optimization(error_function)
    else:
        print(f"\nRunning STANDARD optimization with 4 parameters using {error_function.upper()}...")
        print("(Use 'python optimize_params.py extended' for 6-parameter optimization)")
        print("(Use 'python optimize_params.py mse' for MSE optimization)")
        print("(Use 'python optimize_params.py extended mse' for both)")
        best_params, results = main_optimized(error_function)
    
    if best_params is not None:
        print(f"\nOptimization completed successfully!")
        print(f"Number of parameters optimized: {len(best_params)}")
        print(f"Error function used: {error_function.upper()}")
    else:
        print("\nOptimization failed. Please check your configuration.")