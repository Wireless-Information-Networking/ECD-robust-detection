#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: threshold.py
Author: Javier del Río
Date: 2025-11-14
Description: 
    Find optimal threshold for phase_std to classify static vs dynamic arcs.
    Uses gradient descent with Mean Relative Error (MRE) as cost function.

License: MIT License
Dependencies: numpy, sklearn, feature_extraction, data_loader, arc_segmentation
"""

import numpy as np
from feature_extraction import extract_features
from data_loader import load_dataset_from_config
from config_manager import load_optimized_parameters
from arc_segmentation import segment_tag_data_into_arcs
from scipy.optimize import minimize_scalar

def mre_cost_function(threshold, phase_std_values, labels):
    """
    Mean Relative Error cost function for threshold optimization.
    
    :param threshold: Current threshold value
    :param phase_std_values: Array of phase_std values
    :param labels: True labels (0=Static, 1=Dynamic)
    :return: Mean relative error
    """
    # Classify based on threshold: phase_std < threshold → Static (0), else → Dynamic (1)
    predictions = (phase_std_values >= threshold).astype(int)
    
    # Calculate error
    errors = np.abs(predictions - labels)
    mre = np.mean(errors)
    
    return mre

def compute_gradient(threshold, phase_std_values, labels, epsilon=1e-4):
    """
    Compute numerical gradient of MRE cost function.
    
    :param threshold: Current threshold value
    :param phase_std_values: Array of phase_std values
    :param labels: True labels
    :param epsilon: Small value for numerical differentiation
    :return: Gradient value
    """
    # Numerical gradient using central difference
    cost_plus = mre_cost_function(threshold + epsilon, phase_std_values, labels)
    cost_minus = mre_cost_function(threshold - epsilon, phase_std_values, labels)
    
    gradient = (cost_plus - cost_minus) / (2 * epsilon)
    
    return gradient

def find_optimal_threshold_gradient_descent(phase_std_values, labels, 
                                           initial_threshold=None,
                                           learning_rate=0.01,
                                           max_iterations=1000,
                                           tolerance=1e-6,
                                           verbose=True):
    """
    Find optimal threshold using gradient descent with MRE cost function.
    
    :param phase_std_values: Array of phase_std feature values
    :param labels: True labels (0=Static, 1=Dynamic)
    :param initial_threshold: Starting threshold (if None, use median)
    :param learning_rate: Learning rate for gradient descent
    :param max_iterations: Maximum number of iterations
    :param tolerance: Convergence tolerance
    :param verbose: Print iteration details
    :return: Optimal threshold and history
    """
    print(f"\n{'='*60}")
    print("GRADIENT DESCENT THRESHOLD OPTIMIZATION")
    print("="*60)
    
    # Initialize threshold
    if initial_threshold is None:
        initial_threshold = np.median(phase_std_values)
    
    threshold = initial_threshold
    
    print(f"\nInitial threshold: {threshold:.4f}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max iterations: {max_iterations}")
    print(f"Tolerance: {tolerance}")
    
    # Track history
    history = {
        'thresholds': [threshold],
        'costs': [mre_cost_function(threshold, phase_std_values, labels)],
        'gradients': []
    }
    
    print(f"\nStarting optimization...")
    print(f"{'Iteration':<10} {'Threshold':<12} {'MRE Cost':<12} {'Gradient':<12}")
    print("-" * 50)
    
    for iteration in range(max_iterations):
        # Compute gradient
        gradient = compute_gradient(threshold, phase_std_values, labels)
        history['gradients'].append(gradient)
        
        # Update threshold (gradient descent)
        threshold_new = threshold - learning_rate * gradient
        
        # Ensure threshold stays within valid range
        threshold_new = np.clip(threshold_new, 
                               phase_std_values.min(), 
                               phase_std_values.max())
        
        # Compute new cost
        cost_new = mre_cost_function(threshold_new, phase_std_values, labels)
        
        # Store history
        history['thresholds'].append(threshold_new)
        history['costs'].append(cost_new)
        
        # Print progress
        if verbose and (iteration % 10 == 0 or iteration < 10):
            print(f"{iteration:<10} {threshold_new:<12.4f} {cost_new:<12.4f} {gradient:<12.6f}")
        
        # Check convergence
        if abs(threshold_new - threshold) < tolerance:
            print(f"\nConverged at iteration {iteration}")
            threshold = threshold_new
            break
        
        threshold = threshold_new
    
    final_cost = mre_cost_function(threshold, phase_std_values, labels)
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Optimal threshold: {threshold:.4f}")
    print(f"Final MRE cost: {final_cost:.4f}")
    print(f"Total iterations: {len(history['thresholds']) - 1}")
    
    return threshold, history

def find_optimal_threshold_grid_search(phase_std_values, labels, n_points=1000):
    """
    Find optimal threshold using exhaustive grid search (for comparison).
    
    :param phase_std_values: Array of phase_std feature values
    :param labels: True labels
    :param n_points: Number of grid points to evaluate
    :return: Optimal threshold
    """
    print(f"\n{'='*60}")
    print("GRID SEARCH THRESHOLD OPTIMIZATION (for comparison)")
    print("="*60)
    
    # Create grid of threshold values
    thresholds = np.linspace(phase_std_values.min(), 
                            phase_std_values.max(), 
                            n_points)
    
    # Evaluate MRE for each threshold
    costs = [mre_cost_function(t, phase_std_values, labels) for t in thresholds]
    
    # Find optimal
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs[optimal_idx]
    
    print(f"\nGrid search results:")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Optimal MRE cost: {optimal_cost:.4f}")
    print(f"  Grid points evaluated: {n_points}")
    
    return optimal_threshold, thresholds, costs

def evaluate_threshold(threshold, phase_std_values, labels):
    """
    Evaluate classification performance with given threshold.
    
    :param threshold: Threshold value
    :param phase_std_values: Array of phase_std values
    :param labels: True labels
    """
    print(f"\n{'='*60}")
    print("THRESHOLD EVALUATION")
    print("="*60)
    
    # Make predictions
    predictions = (phase_std_values >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(predictions == labels)
    
    # Confusion matrix
    true_negatives = np.sum((predictions == 0) & (labels == 0))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    true_positives = np.sum((predictions == 1) & (labels == 1))
    
    total_static = np.sum(labels == 0)
    total_dynamic = np.sum(labels == 1)
    
    print(f"\nThreshold: {threshold:.4f}")
    print(f"Classification rule: phase_std >= {threshold:.4f} → Dynamic")
    print(f"                     phase_std <  {threshold:.4f} → Static")
    
    print(f"\n{'='*50}")
    print("PERFORMANCE METRICS")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Mean Relative Error (MRE): {1-accuracy:.4f}")
    
    print(f"\n{'='*50}")
    print("CONFUSION MATRIX")
    print("="*50)
    print(f"{'':20} {'Predicted Static':>18} {'Predicted Dynamic':>18}")
    print(f"{'True Static':<20} {true_negatives:>18} {false_positives:>18}")
    print(f"{'True Dynamic':<20} {false_negatives:>18} {true_positives:>18}")
    
    print(f"\n{'='*50}")
    print("DETAILED STATISTICS")
    print("="*50)
    
    # Metrics per class
    if total_static > 0:
        static_accuracy = true_negatives / total_static
        print(f"\nStatic arcs (Class 0):")
        print(f"  Total: {total_static}")
        print(f"  Correctly classified: {true_negatives} ({static_accuracy*100:.2f}%)")
        print(f"  Misclassified as Dynamic: {false_positives} ({(1-static_accuracy)*100:.2f}%)")
    
    if total_dynamic > 0:
        dynamic_accuracy = true_positives / total_dynamic
        print(f"\nDynamic arcs (Class 1):")
        print(f"  Total: {total_dynamic}")
        print(f"  Correctly classified: {true_positives} ({dynamic_accuracy*100:.2f}%)")
        print(f"  Misclassified as Static: {false_negatives} ({(1-dynamic_accuracy)*100:.2f}%)")
    
    # Precision and Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*50}")
    print("CLASSIFICATION METRICS (for Dynamic class)")
    print("="*50)
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score: {f1_score:.4f}")
    
    return accuracy, predictions

def visualize_threshold_results(phase_std_values, labels, threshold, history=None):
    """
    Visualize threshold optimization results.
    
    :param phase_std_values: Array of phase_std values
    :param labels: True labels
    :param threshold: Optimal threshold
    :param history: Optimization history (optional)
    """
    import matplotlib.pyplot as plt
    
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Distribution of phase_std by class
    ax1 = axes[0, 0]
    static_values = phase_std_values[labels == 0]
    dynamic_values = phase_std_values[labels == 1]
    
    ax1.hist(static_values, bins=50, alpha=0.6, label='Static (Class 0)', 
             color='blue', edgecolor='black')
    ax1.hist(dynamic_values, bins=50, alpha=0.6, label='Dynamic (Class 1)', 
             color='red', edgecolor='black')
    ax1.axvline(threshold, color='green', linestyle='--', linewidth=2,
                label=f'Optimal threshold: {threshold:.4f}')
    ax1.set_xlabel('phase_std', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of phase_std by Class', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot with threshold
    ax2 = axes[0, 1]
    indices = np.arange(len(phase_std_values))
    colors = ['blue' if l == 0 else 'red' for l in labels]
    ax2.scatter(indices, phase_std_values, c=colors, alpha=0.5, s=20)
    ax2.axhline(threshold, color='green', linestyle='--', linewidth=2,
                label=f'Threshold: {threshold:.4f}')
    ax2.set_xlabel('Arc Index', fontsize=12)
    ax2.set_ylabel('phase_std', fontsize=12)
    ax2.set_title('phase_std Values with Classification Threshold', fontsize=14, fontweight='bold')
    ax2.legend(['Static', 'Dynamic', f'Threshold: {threshold:.4f}'])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Optimization history (if available)
    if history is not None:
        ax3 = axes[1, 0]
        iterations = range(len(history['costs']))
        ax3.plot(iterations, history['costs'], 'b-', linewidth=2)
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('MRE Cost', fontsize=12)
        ax3.set_title('Gradient Descent Convergence', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Annotate final value
        final_cost = history['costs'][-1]
        ax3.annotate(f'Final: {final_cost:.4f}', 
                    xy=(len(iterations)-1, final_cost),
                    xytext=(len(iterations)*0.7, final_cost*1.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=10, color='red')
    else:
        axes[1, 0].text(0.5, 0.5, 'No optimization history available',
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Optimization History', fontsize=14, fontweight='bold')
    
    # Plot 4: Classification regions
    ax4 = axes[1, 1]
    
    # Create sorted values for visualization
    sorted_indices = np.argsort(phase_std_values)
    sorted_values = phase_std_values[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Plot regions
    x_range = np.linspace(sorted_values.min(), sorted_values.max(), 1000)
    y_pred = (x_range >= threshold).astype(float)
    
    ax4.fill_between(x_range, 0, y_pred, alpha=0.3, color='red', label='Dynamic region')
    ax4.fill_between(x_range, y_pred, 1, alpha=0.3, color='blue', label='Static region')
    
    # Overlay actual data points
    for i, (val, label) in enumerate(zip(sorted_values, sorted_labels)):
        color = 'blue' if label == 0 else 'red'
        marker = 'o' if label == 0 else '^'
        ax4.scatter(val, label, c=color, marker=marker, s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax4.axvline(threshold, color='green', linestyle='--', linewidth=2,
                label=f'Threshold: {threshold:.4f}')
    ax4.set_xlabel('phase_std', fontsize=12)
    ax4.set_ylabel('Class (0=Static, 1=Dynamic)', fontsize=12)
    ax4.set_title('Classification Regions', fontsize=14, fontweight='bold')
    ax4.set_ylim(-0.1, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    import os
    os.makedirs('output_plots', exist_ok=True)
    plt.savefig('output_plots/threshold_optimization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to: output_plots/threshold_optimization.png")
    plt.show()

def main_threshold_finder():
    """
    Main function to find optimal phase_std threshold for static/dynamic classification.
    """
    print("="*60)
    print("PHASE_STD THRESHOLD FINDER")
    print("Static vs Dynamic Arc Classification")
    print("="*60)
    
    # 1. Load dataset configuration (same as xgboost_model.py)
    dataset_config = {
        'files': [
            {'path': 'data/dynamic.csv', 'label': 1},  # Dynamic
            {'path': 'data/static.csv', 'label': 0},   # Static
            {'path': 'data/magic_mike.csv', 'label': 1}  # Magic (dynamic)
        ],
        'directories': [
            {'path': 'data/2025-07-15/', 'label': 1},
            {'path': 'data/2025-07-22/', 'label': 1},
            {'path': 'data/2025-07-25/', 'label': 1},
            {'path': 'data/2025-07-28/', 'label': 1},
            {'path': 'data/2025-07-31/', 'label': 1},
            {'path': 'data/2025-10-07/', 'label': 0},
        ]
    }
    
    # 2. Load optimized segmentation parameters
    print("\nLoading optimized segmentation parameters...")
    try:
        optimized_params = load_optimized_parameters('output_data/extended_optimized_parameters.json')
        segmentation_params = {
            'abs_threshold': optimized_params.get('abs_threshold', 1.0),
            'stat_threshold': optimized_params.get('stat_threshold', 2.0),
            'num_interp_points': int(optimized_params.get('num_interp_points', 200)),
            'smoothing_sigma': optimized_params.get('smoothing_sigma', 2.0),
            'min_arc_duration': optimized_params.get('min_arc_duration', 0.1),
            'min_arc_samples': optimized_params.get('min_arc_samples', 5),
            'minima_min_distance': optimized_params.get('minima_min_distance', 10),
            'minima_prominence': optimized_params.get('minima_prominence', 0.1)
        }
        print("✓ Loaded optimized parameters")
    except Exception as e:
        print(f"⚠ Could not load optimized parameters: {e}")
        print("Using default parameters...")
        segmentation_params = {
            'abs_threshold': 1.0,
            'stat_threshold': 2.0,
            'num_interp_points': 200,
            'smoothing_sigma': 2.0,
            'min_arc_duration': 0.1,
            'min_arc_samples': 5,
            'minima_min_distance': 10,
            'minima_prominence': 0.1
        }
    
    # 3. Load dataset
    print("\nLoading dataset...")
    tag_data_list, labels = load_dataset_from_config(dataset_config)
    print(f"✓ Loaded {len(tag_data_list)} tags")
    
    # 4. Segment data into arcs
    print(f"\n{'='*60}")
    print("ARC SEGMENTATION")
    print("="*60)
    
    all_arcs = []
    all_arc_labels = []
    
    for i, (tag_data, label) in enumerate(zip(tag_data_list, labels)):
        try:
            for key in ['timestamp', 'rssi', 'phase']:
                if key in tag_data:
                    tag_data[key] = np.array(tag_data[key])
            
            arcs = segment_tag_data_into_arcs(
                tag_data,
                abs_threshold=segmentation_params['abs_threshold'],
                stat_threshold=segmentation_params['stat_threshold'],
                num_interp_points=segmentation_params['num_interp_points'],
                smoothing_sigma=segmentation_params['smoothing_sigma'],
                min_arc_duration=segmentation_params['min_arc_duration'],
                min_arc_samples=segmentation_params['min_arc_samples'],
                minima_min_distance=segmentation_params['minima_min_distance'],
                minima_prominence=segmentation_params['minima_prominence'],
                verbose=False
            )
            
            arc_labels = [label] * len(arcs)
            all_arcs.extend(arcs)
            all_arc_labels.extend(arc_labels)
            
        except Exception as e:
            print(f"  Error processing tag {i+1}: {e}")
            continue
    
    print(f"✓ Total arcs created: {len(all_arcs)}")
    unique_labels, counts = np.unique(all_arc_labels, return_counts=True)
    print(f"  Arc distribution: {dict(zip(['Static', 'Dynamic'], counts))}")
    
    # 5. Extract phase_std feature from all arcs
    print(f"\n{'='*60}")
    print("FEATURE EXTRACTION (phase_std only)")
    print("="*60)
    
    phase_std_values = []
    valid_labels = []
    
    for i, arc in enumerate(all_arcs):
        try:
            arc_segment = {
                'timestamp': arc['timestamp'],
                'rssi': arc['rssi'],
                'phase': arc['phase']
            }
            
            features_dict = extract_features(arc_segment)
            phase_std_values.append(features_dict['phase_std'])
            valid_labels.append(all_arc_labels[i])
            
        except Exception as e:
            print(f"  Error extracting features from arc {i+1}: {e}")
            continue
    
    phase_std_values = np.array(phase_std_values)
    valid_labels = np.array(valid_labels)
    
    print(f"✓ Extracted phase_std from {len(phase_std_values)} arcs")
    print(f"\nphase_std statistics:")
    print(f"  Range: [{phase_std_values.min():.4f}, {phase_std_values.max():.4f}]")
    print(f"  Mean: {phase_std_values.mean():.4f}")
    print(f"  Median: {np.median(phase_std_values):.4f}")
    print(f"  Std: {phase_std_values.std():.4f}")
    
    print(f"\nphase_std by class:")
    for label_val, label_name in [(0, 'Static'), (1, 'Dynamic')]:
        class_values = phase_std_values[valid_labels == label_val]
        if len(class_values) > 0:
            print(f"  {label_name}:")
            print(f"    Count: {len(class_values)}")
            print(f"    Range: [{class_values.min():.4f}, {class_values.max():.4f}]")
            print(f"    Mean: {class_values.mean():.4f}")
            print(f"    Median: {np.median(class_values):.4f}")
    
    # 6. Find optimal threshold using gradient descent
    optimal_threshold_gd, history = find_optimal_threshold_gradient_descent(
        phase_std_values, 
        valid_labels,
        learning_rate=0.1,
        max_iterations=1000,
        verbose=True
    )
    
    # 7. Find optimal threshold using grid search (for comparison)
    optimal_threshold_grid, thresholds_grid, costs_grid = find_optimal_threshold_grid_search(
        phase_std_values,
        valid_labels,
        n_points=1000
    )
    
    # 8. Compare methods
    print(f"\n{'='*60}")
    print("METHOD COMPARISON")
    print("="*60)
    print(f"Gradient Descent:")
    print(f"  Optimal threshold: {optimal_threshold_gd:.4f}")
    print(f"  MRE cost: {mre_cost_function(optimal_threshold_gd, phase_std_values, valid_labels):.4f}")
    print(f"\nGrid Search:")
    print(f"  Optimal threshold: {optimal_threshold_grid:.4f}")
    print(f"  MRE cost: {mre_cost_function(optimal_threshold_grid, phase_std_values, valid_labels):.4f}")
    print(f"\nDifference: {abs(optimal_threshold_gd - optimal_threshold_grid):.4f}")
    
    # 9. Use gradient descent result as final
    optimal_threshold = optimal_threshold_gd
    
    # 10. Evaluate final threshold
    accuracy, predictions = evaluate_threshold(optimal_threshold, phase_std_values, valid_labels)
    
    # 11. Visualize results
    visualize_threshold_results(phase_std_values, valid_labels, optimal_threshold, history)
    
    # 12. Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print("="*60)
    
    import os
    import json
    os.makedirs('output_data', exist_ok=True)
    
    results = {
        'optimal_threshold': float(optimal_threshold),
        'accuracy': float(accuracy),
        'mre_cost': float(mre_cost_function(optimal_threshold, phase_std_values, valid_labels)),
        'method': 'gradient_descent',
        'total_arcs': int(len(phase_std_values)),
        'static_arcs': int(np.sum(valid_labels == 0)),
        'dynamic_arcs': int(np.sum(valid_labels == 1)),
        'phase_std_range': [float(phase_std_values.min()), float(phase_std_values.max())],
        'segmentation_params': segmentation_params,
        'classification_rule': f'phase_std >= {optimal_threshold:.4f} → Dynamic, phase_std < {optimal_threshold:.4f} → Static'
    }
    
    with open('output_data/threshold_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Results saved to: output_data/threshold_results.json")
    
    # 13. Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\n🎯 OPTIMAL THRESHOLD FOUND: {optimal_threshold:.4f}")
    print(f"\n📊 Classification Rule:")
    print(f"   IF phase_std >= {optimal_threshold:.4f}")
    print(f"      → Arc is DYNAMIC")
    print(f"   ELSE (phase_std < {optimal_threshold:.4f})")
    print(f"      → Arc is STATIC")
    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📉 MRE Cost: {1-accuracy:.4f}")
    print(f"\n💾 Results saved to:")
    print(f"   - output_data/threshold_results.json")
    print(f"   - output_plots/threshold_optimization.png")
    
    return optimal_threshold, accuracy, results

def accuracy_cost_function(threshold, phase_std_values, labels):
    """
    Direct accuracy-based cost function (to maximize accuracy, we minimize 1-accuracy).
    
    :param threshold: Current threshold value
    :param phase_std_values: Array of phase_std values
    :param labels: True labels (0=Static, 1=Dynamic)
    :return: 1 - accuracy (to minimize)
    """
    predictions = (phase_std_values >= threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    return 1 - accuracy  # We want to minimize this

def weighted_cost_function(threshold, phase_std_values, labels, weight_dynamic=1.0):
    """
    Weighted cost function to handle class imbalance or prioritize one class.
    
    :param weight_dynamic: Weight for dynamic class errors (increase to prioritize dynamic detection)
    """
    predictions = (phase_std_values >= threshold).astype(int)
    
    # Separate errors by class
    static_errors = np.sum((predictions == 1) & (labels == 0))  # False positives
    dynamic_errors = np.sum((predictions == 0) & (labels == 1))  # False negatives
    
    # Weighted error
    total_error = static_errors + weight_dynamic * dynamic_errors
    total_samples = len(labels)
    
    return total_error / total_samples

def f1_cost_function(threshold, phase_std_values, labels):
    """
    Cost function based on F1-score (to maximize F1, minimize 1-F1).
    """
    predictions = (phase_std_values >= threshold).astype(int)
    
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return 1 - f1_score  # Minimize to maximize F1

if __name__ == "__main__":
    optimal_threshold, accuracy, results = main_threshold_finder()