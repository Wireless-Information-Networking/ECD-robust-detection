#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: xgboost_model.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    XGBoost classifier implementation for RFID tag presence detection with GPU acceleration.
    Processes RFID tag data to classify dynamic vs static scenarios using
    extracted statistical features with cross-validation and comprehensive
    performance evaluation utilizing CUDA for enhanced performance.

License: MIT License
Dependencies: xgboost, sklearn, cupy, feature_extraction (local), data_loader (local), timediff (local)
"""

# filepath: /home/javier/Documents/CFD-rfid-cleaner/src/xgboost_model.py

from xgboost import XGBClassifier, DMatrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cupy
import numpy as np
import joblib

from feature_extraction import extract_features, normalize_features
from timediff import split_tag_data_by_time
from data_loader import load_dataset_from_config, print_dataset_summary, validate_dataset

def prepare_segments_and_labels(tag_data_list, labels, time_threshold=5.0):
    """
    Split tag data into time segments and prepare labels for each segment.
    
    :param tag_data_list: List of tag data dictionaries
    :param labels: List of labels for each tag
    :param time_threshold: Time threshold for segmentation
    :return: Tuple of (segments, segment_labels)
    """
    all_segments = []
    all_segment_labels = []
    
    print(f"Segmenting {len(tag_data_list)} tags with time threshold {time_threshold}s...")
    
    for i, (tag_data, label) in enumerate(zip(tag_data_list, labels)):
        try:
            # Split tag data by time segments
            segments = split_tag_data_by_time(tag_data, time_threshold)
            
            # Assign the same label to all segments from this tag
            segment_labels = [label] * len(segments)
            
            all_segments.extend(segments)
            all_segment_labels.extend(segment_labels)
            
            print(f"  Tag {i+1}: {len(segments)} segments (label {label})")
            
        except Exception as e:
            print(f"  Error processing tag {i+1}: {e}")
            continue
    
    print(f"Total segments created: {len(all_segments)}")
    unique_labels, counts = np.unique(all_segment_labels, return_counts=True)
    print(f"Segment label distribution: {dict(zip(unique_labels, counts))}")
    
    return all_segments, all_segment_labels

def prepare_arcs_and_labels(tag_data_list, labels, segmentation_params):
    """
    Segment tag data into arcs using advanced arc segmentation and prepare labels for each arc.
    
    :param tag_data_list: List of tag data dictionaries
    :param labels: List of labels for each tag
    :param segmentation_params: Dictionary with segmentation parameters
    :return: Tuple of (arcs, arc_labels)
    """
    from arc_segmentation import segment_tag_data_into_arcs
    
    all_arcs = []
    all_arc_labels = []
    
    print(f"Segmenting {len(tag_data_list)} tags using advanced arc segmentation...")
    print(f"Parameters: abs_threshold={segmentation_params['abs_threshold']:.3f}, "
          f"stat_threshold={segmentation_params['stat_threshold']:.3f}, "
          f"num_interp_points={segmentation_params['num_interp_points']}, "
          f"smoothing_sigma={segmentation_params['smoothing_sigma']:.3f}")
    
    for i, (tag_data, label) in enumerate(zip(tag_data_list, labels)):
        try:
            # Convert to numpy arrays if needed
            for key in ['timestamp', 'rssi', 'phase']:
                if key in tag_data:
                    tag_data[key] = np.array(tag_data[key])
            
            # Use advanced arc segmentation
            arcs = segment_tag_data_into_arcs(
                tag_data,
                abs_threshold=segmentation_params['abs_threshold'],
                stat_threshold=segmentation_params['stat_threshold'],
                num_interp_points=segmentation_params['num_interp_points'],
                smoothing_sigma=segmentation_params['smoothing_sigma'],
                min_arc_duration=segmentation_params.get('min_arc_duration', 0.1),
                min_arc_samples=segmentation_params.get('min_arc_samples', 5),
                minima_min_distance=segmentation_params.get('minima_min_distance', 10),
                minima_prominence=segmentation_params.get('minima_prominence', 0.1),
                verbose=False  # Suppress detailed output for ML processing
            )
            
            # Assign the same label to all arcs from this tag
            arc_labels = [label] * len(arcs)
            
            all_arcs.extend(arcs)
            all_arc_labels.extend(arc_labels)
            
            print(f"  Tag {i+1}: {len(arcs)} arcs detected (label {label})")
            
        except Exception as e:
            print(f"  Error processing tag {i+1}: {e}")
            continue
    
    print(f"Total arcs created: {len(all_arcs)}")
    unique_labels, counts = np.unique(all_arc_labels, return_counts=True)
    print(f"Arc label distribution: {dict(zip(unique_labels, counts))}")
    
    return all_arcs, all_arc_labels

def analyze_decision_thresholds(clf, feature_names, top_n_trees=5):
    """
    Analyze and display decision thresholds from XGBoost trees.
    Simple method to understand how the model makes decisions.
    
    :param clf: Trained XGBoost classifier
    :param feature_names: List of feature names
    :param top_n_trees: Number of trees to analyze in detail
    """
    print(f"\n{'='*60}")
    print("DECISION TREE CRITERIA ANALYSIS")
    print("="*60)
    
    booster = clf.get_booster()
    trees_df = booster.trees_to_dataframe()
    
    # Filter only split nodes (not leaf nodes)
    split_nodes = trees_df[trees_df['Feature'].notna() & (trees_df['Feature'] != 'Leaf')]
    
    if len(split_nodes) == 0:
        print("No split nodes found in the trees.")
        return
    
    print(f"\nTotal trees in model: {clf.n_estimators}")
    print(f"Total split nodes across all trees: {len(split_nodes)}")
    
    # 1. Summary of splits per feature
    print(f"\n{'='*50}")
    print("1. SPLITS PER FEATURE")
    print("="*50)
    
    for i, feature_name in enumerate(feature_names):
        feature_key = f'f{i}'
        feature_splits = split_nodes[split_nodes['Feature'] == feature_key]
        
        if len(feature_splits) > 0:
            splits = feature_splits['Split'].values
            print(f"\n{feature_name}:")
            print(f"  Number of splits: {len(splits)}")
            print(f"  Split range: [{splits.min():.4f}, {splits.max():.4f}]")
            print(f"  Mean split: {splits.mean():.4f}")
            print(f"  Median split: {np.median(splits):.4f}")
            print(f"  Most common splits (top 5):")
            unique, counts = np.unique(np.round(splits, 3), return_counts=True)
            top_indices = np.argsort(counts)[-5:][::-1]
            for idx in top_indices:
                print(f"    {unique[idx]:.4f} (used {counts[idx]} times)")
    
    # 2. Detailed analysis of first few trees
    print(f"\n{'='*50}")
    print(f"2. DETAILED TREE ANALYSIS (First {top_n_trees} trees)")
    print("="*50)
    
    for tree_id in range(min(top_n_trees, clf.n_estimators)):
        tree_nodes = trees_df[trees_df['Tree'] == tree_id]
        tree_splits = tree_nodes[tree_nodes['Feature'].notna() & (tree_nodes['Feature'] != 'Leaf')]
        
        print(f"\n--- Tree {tree_id} ---")
        print(f"Total nodes: {len(tree_nodes)}")
        print(f"Split nodes: {len(tree_splits)}")
        print(f"Leaf nodes: {len(tree_nodes) - len(tree_splits)}")
        
        if len(tree_splits) > 0:
            print(f"\nDecision rules:")
            for _, node in tree_splits.iterrows():
                feature_idx = int(node['Feature'][1:])  # Extract number from 'f0', 'f1', etc.
                feature_name = feature_names[feature_idx]
                split_value = node['Split']
                print(f"  Node {node['Node']}: IF {feature_name} < {split_value:.4f}")
    
    # 3. If only one feature, show detailed decision boundary
    if len(feature_names) == 1:
        print(f"\n{'='*50}")
        print("3. SINGLE FEATURE DECISION ANALYSIS")
        print("="*50)
        
        feature_key = 'f0'
        feature_splits = split_nodes[split_nodes['Feature'] == feature_key]['Split'].values
        
        if len(feature_splits) > 0:
            print(f"\nFeature: {feature_names[0]}")
            print(f"All split points: {sorted(np.unique(np.round(feature_splits, 4)))}")
            print(f"\nPrimary decision threshold (median): {np.median(feature_splits):.4f}")
            print(f"\nDecision rule interpretation:")
            median_split = np.median(feature_splits)
            print(f"  IF {feature_names[0]} < {median_split:.4f}")
            print(f"    → TENDS TO: Class 0 (Static)")
            print(f"  ELSE ({feature_names[0]} >= {median_split:.4f})")
            print(f"    → TENDS TO: Class 1 (Dynamic)")
            
            # Distribution of splits
            print(f"\nSplit distribution:")
            percentiles = [10, 25, 50, 75, 90]
            for p in percentiles:
                val = np.percentile(feature_splits, p)
                print(f"  {p}th percentile: {val:.4f}")
    
    return trees_df

def visualize_single_feature_decision(clf, X, y, feature_name, scaler=None):
    """
    Visualize decision boundary for single feature model.
    
    :param clf: Trained XGBoost classifier
    :param X: Feature data (normalized)
    :param y: Labels
    :param feature_name: Name of the feature
    :param scaler: Scaler used for normalization (optional)
    """
    import matplotlib.pyplot as plt
    
    if X.shape[1] != 1:
        print("This visualization is only for single-feature models.")
        return
    
    print(f"\n{'='*60}")
    print("SINGLE FEATURE DECISION VISUALIZATION")
    print("="*60)
    
    # Create range of values
    X_flat = X.flatten()
    x_min, x_max = X_flat.min(), X_flat.max()
    x_range = np.linspace(x_min - 0.5, x_max + 0.5, 1000).reshape(-1, 1)
    
    # Get predictions
    y_prob = clf.predict_proba(x_range)[:, 1]
    y_pred_class = clf.predict(x_range)
    
    # Find decision boundary (where prob = 0.5)
    threshold_idx = np.argmin(np.abs(y_prob - 0.5))
    threshold_value = x_range[threshold_idx, 0]
    
    print(f"\nDecision threshold at P=0.5: {threshold_value:.4f}")
    
    # If scaler provided, show original scale
    if scaler is not None:
        # Inverse transform the threshold
        threshold_original = scaler.inverse_transform([[threshold_value]])[0, 0]
        print(f"Original scale threshold: {threshold_original:.4f}")
         
        
    return threshold_value

def main_xgboost_training():
    """
    Main function for XGBoost training with configurable data loading and optimized arc segmentation.
    """
    from config_manager import load_optimized_parameters
    
    print("=== XGBOOST CLASSIFIER FOR RFID TAG ANALYSIS ===\n")
    
    # Configuration for data loading
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
    
    # Load optimized segmentation parameters
    print("Loading optimized segmentation parameters...")
    try:
        optimized_params = load_optimized_parameters('output_data/extended_optimized_parameters.json')
        
        # Create segmentation configuration from optimized parameters
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
        
        print("✓ Loaded optimized parameters:")
        for key, value in segmentation_params.items():
            print(f"  {key}: {value}")
        
        if 'optimization_date' in optimized_params:
            print(f"  Optimized on: {optimized_params['optimization_date']}")
            
    except Exception as e:
        print(f"⚠ Could not load optimized parameters: {e}")
        print("Using default segmentation parameters...")
        
        # Default parameters as fallback
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
    
    # Load dataset using data_loader functions
    print("\nLoading dataset...")
    tag_data_list, labels = load_dataset_from_config(dataset_config)
    
    if not tag_data_list:
        print("No data loaded. Please check file paths and configuration.")
        return
    
    # Validate dataset
    if not validate_dataset(tag_data_list, labels):
        print("Dataset validation failed. Please check your data.")
        return
    
    # Print dataset summary
    print_dataset_summary(tag_data_list, labels)
    
    # Segment data into arcs using optimized parameters
    print(f"\n{'='*60}")
    print("ARC SEGMENTATION PHASE")
    print("="*60)
    
    arcs, arc_labels = prepare_arcs_and_labels(tag_data_list, labels, segmentation_params)
    
    if not arcs:
        print("No arcs created. Please check segmentation parameters.")
        return
    
    # Extract features from each arc
    print(f"\n{'='*60}")
    print("FEATURE EXTRACTION PHASE")
    print("="*60)
    print(f"Extracting features from {len(arcs)} arcs...")
    
    features_list = []
    valid_labels = []
    feature_names_descriptive = None
    
    for i, arc in enumerate(arcs):
        try:
            # Create segment-like dictionary for feature extraction compatibility
            arc_segment = {
                'timestamp': arc['timestamp'],
                'rssi': arc['rssi'],
                'phase': arc['phase']
            }
            
            # Extract features (returns dictionary with descriptive keys)
            features_dict = extract_features(arc_segment)
            
            # Get feature names from the first successful extraction
            if feature_names_descriptive is None:
                feature_names_descriptive = list(features_dict.keys())
                print(f"Detected {len(feature_names_descriptive)} features: {feature_names_descriptive}")
            
            # Convert dictionary to list of values in consistent order
            features_values = [features_dict[key] for key in feature_names_descriptive]
            
            features_list.append(features_values)
            valid_labels.append(arc_labels[i])
            
        except Exception as e:
            print(f"  Error extracting features from arc {i+1}: {e}")
            continue
    
    print(f"Features extracted from {len(features_list)} arcs")
    
    if not features_list:
        print("No features extracted. Please check your data and feature extraction.")
        return
    
    # Remove features with low variance or that are not useful for classification
    print(f"\n{'='*60}")
    print("FEATURE SELECTION PHASE")
    print("="*60)
    
    # Specify which features to remove (example)
    features_to_remove = ['total_time', 'rssi_max', 'time_diff_max', 'time_diff_std', 'num_samples', 'time_diff_min', 'rssi_std', 'time_diff_mean', 'phase_mean', 'rssi_mean', 'rssi_min'] # Cambia según lo que quieras probar
    # features_to_remove = []  # Para no remover ninguna
    
    # Remove specified features
    filtered_features_list, filtered_feature_names, removal_info = remove_features(
        features_list, 
        feature_names_descriptive, 
        features_to_remove
    )
    
    # Update variables to use filtered data
    features_list = filtered_features_list
    feature_names_descriptive = filtered_feature_names
    
    print(f"Features after removal: {len(features_list)} arcs, {len(feature_names_descriptive)} features")
    
    # Normalize features (ahora con las features filtradas)
    print("\nNormalizing features...")
    features_normalized, scaler = normalize_features(features_list)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features_normalized, valid_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=valid_labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train XGBoost model with GPU acceleration
    print(f"\n{'='*60}")
    print("MODEL TRAINING PHASE")
    print("="*60)
    print("Training XGBoost model with GPU acceleration...")
    
    clf = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist',
        device='cuda',
        random_state=42
    )
    
    clf.fit(cupy.array(X_train), y_train)
    print("Training completed!")
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print("="*60)
    y_pred = clf.predict(cupy.array(X_test))
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Static', 'Dynamic']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 5-fold cross-validation
    print("\n=== CROSS-VALIDATION ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, cupy.array(features_normalized), valid_labels, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Classification report for cross-validation
    y_pred_cv = cross_val_predict(clf, cupy.array(features_normalized), valid_labels, cv=cv)
    print("\nCross-validation Classification Report:")
    print(classification_report(valid_labels, y_pred_cv, target_names=['Static', 'Dynamic']))
    
    # Feature importance with descriptive names from extract_features()
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    feature_importance = clf.feature_importances_
    
    # Create pairs of (feature_name, importance) using the descriptive names from extract_features
    importance_pairs = list(zip(feature_names_descriptive, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("Top features by importance:")
    print("-" * 50)
    for i, (name, importance) in enumerate(importance_pairs):
        print(f"  {i+1:2d}. {name:<20}: {importance:.4f}")
    
    # Feature importance analysis by category (based on feature names from extract_features)
    print(f"\nFeature importance by category:")
    print("-" * 50)
    
    # Group features by prefix (rssi_, phase_, time_, etc.)
    rssi_features = [pair for pair in importance_pairs if pair[0].startswith('rssi_')]
    phase_features = [pair for pair in importance_pairs if pair[0].startswith('phase_')]
    time_features = [pair for pair in importance_pairs if any(keyword in pair[0] for keyword in ['time_', 'total_time', 'num_samples'])]
    
    if rssi_features:
        rssi_total = sum(importance for _, importance in rssi_features)
        rssi_avg = rssi_total / len(rssi_features)
        print(f"  RSSI features ({len(rssi_features)} total):")
        print(f"    Total importance: {rssi_total:.4f}")
        print(f"    Average importance: {rssi_avg:.4f}")
        print(f"    Most important: {rssi_features[0][0]} ({rssi_features[0][1]:.4f})")
        
        print(f"    Individual RSSI features:")
        for name, importance in rssi_features:
            print(f"      - {name}: {importance:.4f}")
    
    if phase_features:
        phase_total = sum(importance for _, importance in phase_features)
        phase_avg = phase_total / len(phase_features)
        print(f"\n  Phase features ({len(phase_features)} total):")
        print(f"    Total importance: {phase_total:.4f}")
        print(f"    Average importance: {phase_avg:.4f}")
        print(f"    Most important: {phase_features[0][0]} ({phase_features[0][1]:.4f})")
        
        print(f"    Individual Phase features:")
        for name, importance in phase_features:
            print(f"      - {name}: {importance:.4f}")
    
    if time_features:
        time_total = sum(importance for _, importance in time_features)
        time_avg = time_total / len(time_features)
        print(f"\n  Temporal features ({len(time_features)} total):")
        print(f"    Total importance: {time_total:.4f}")
        print(f"    Average importance: {time_avg:.4f}")
        print(f"    Most important: {time_features[0][0]} ({time_features[0][1]:.4f})")
        
        print(f"    Individual Temporal features:")
        for name, importance in time_features:
            print(f"      - {name}: {importance:.4f}")
    
    # NEW: Analyze decision thresholds from trees
    trees_df = analyze_decision_thresholds(clf, feature_names_descriptive, top_n_trees=3)
    
    # NEW: If single feature model, create detailed visualization
    if len(feature_names_descriptive) == 1:
        import os
        os.makedirs('output_plots', exist_ok=True)
        threshold = visualize_single_feature_decision(
            clf, 
            features_normalized, 
            valid_labels, 
            feature_names_descriptive[0],
            scaler
        )
    
    # Save model with feature names
    print(f"\n{'='*60}")
    print("SAVING MODEL AND METADATA")
    print("="*60)
    
    import os
    import json
    os.makedirs('output_data', exist_ok=True)
    
    # Save XGBoost model
    clf.save_model('output_data/xgboost_model.json')
    
    # Save scaler
    joblib.dump(scaler, 'output_data/xgboost_scaler.pkl')
    
    # NEW: Save tree criteria if available
    tree_metadata = {}
    if len(feature_names_descriptive) == 1:
        booster = clf.get_booster()
        trees_df_full = booster.trees_to_dataframe()
        split_nodes = trees_df_full[trees_df_full['Feature'] == 'f0']
        if len(split_nodes) > 0:
            splits = split_nodes['Split'].values
            tree_metadata = {
                'single_feature': feature_names_descriptive[0],
                'split_count': int(len(splits)),
                'split_range': [float(splits.min()), float(splits.max())],
                'split_mean': float(splits.mean()),
                'split_median': float(np.median(splits)),
                'decision_threshold': float(np.median(splits))
            }
    
    # Save feature names and importance
    feature_metadata = {
        'feature_names': feature_names_descriptive,
        'feature_importance': {
            name: float(importance) 
            for name, importance in zip(feature_names_descriptive, feature_importance)
        },
        'feature_importance_sorted': [
            {'name': name, 'importance': float(importance)} 
            for name, importance in importance_pairs
        ],
        'model_performance': {
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'test_accuracy': float(test_accuracy)
        },
        'segmentation_params': segmentation_params,
        'tree_criteria': tree_metadata  # NEW: Add tree decision criteria
    }
    
    with open('output_data/xgboost_feature_metadata.json', 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    
    print("✓ Model saved to: output_data/xgboost_model.json")
    print("✓ Scaler saved to: output_data/xgboost_scaler.pkl")
    print("✓ Feature metadata saved to: output_data/xgboost_feature_metadata.json")
    
    # Summary of segmentation performance
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Original tags: {len(tag_data_list)}")
    print(f"Arcs detected: {len(arcs)}")
    print(f"Valid arcs processed: {len(valid_labels)}")
    print(f"Arcs per tag (avg): {len(arcs)/len(tag_data_list):.2f}")
    print(f"Features extracted per arc: {len(feature_names_descriptive)}")
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    print(f"\nMost important features for classification:")
    for i, (name, importance) in enumerate(importance_pairs[:5]):
        print(f"  {i+1}. {name}: {importance:.4f}")
    
    print(f"\nSegmentation method: Advanced arc segmentation")
    print("Parameters used:")
    for key, value in segmentation_params.items():
        print(f"  {key}: {value}")
    
    return clf, scaler, cv_scores.mean(), segmentation_params

def remove_features(features_list, feature_names, features_to_remove):
    """
    Remove specified features from the feature list and update feature names.
    
    :param features_list: List of feature arrays (each row is a sample)
    :param feature_names: List of feature names (descriptive names from extract_features)
    :param features_to_remove: List of feature names to remove (e.g., ['rssi_mean', 'phase_std'])
    :return: Tuple of (filtered_features_list, filtered_feature_names, removed_features_info)
    """
    if not features_to_remove:
        print("No features to remove. Returning original data.")
        return features_list, feature_names, {}
    
    print(f"\n{'='*60}")
    print("FEATURE REMOVAL")
    print("="*60)
    
    # Convert to numpy array for easier manipulation
    features_array = np.array(features_list)
    original_feature_count = len(feature_names)
    
    print(f"Original features ({original_feature_count}): {feature_names}")
    print(f"Removing features: {features_to_remove}")
    
    # Find indices of features to remove
    indices_to_remove = []
    removed_features_info = {}
    
    for feature_to_remove in features_to_remove:
        if feature_to_remove in feature_names:
            idx = feature_names.index(feature_to_remove)
            indices_to_remove.append(idx)
            removed_features_info[feature_to_remove] = {
                'original_index': idx,
                'removed': True
            }
            print(f"  ✓ Found '{feature_to_remove}' at index {idx}")
        else:
            print(f"  ⚠ Feature '{feature_to_remove}' not found in feature list")
            removed_features_info[feature_to_remove] = {
                'original_index': None,
                'removed': False
            }
    
    if not indices_to_remove:
        print("No valid features found to remove. Returning original data.")
        return features_list, feature_names, removed_features_info
    
    # Sort indices in descending order to remove from end to beginning
    indices_to_remove.sort(reverse=True)
    
    # Remove features from array (remove columns)
    filtered_features_array = np.delete(features_array, indices_to_remove, axis=1)
    
    # Remove feature names
    filtered_feature_names = [name for i, name in enumerate(feature_names) 
                             if i not in indices_to_remove]
    
    # Convert back to list format
    filtered_features_list = filtered_features_array.tolist()
    
    print(f"\nFeature removal completed:")
    print(f"  Original feature count: {original_feature_count}")
    print(f"  Features removed: {len(indices_to_remove)}")
    print(f"  Final feature count: {len(filtered_feature_names)}")
    print(f"  Remaining features: {filtered_feature_names}")
    
    # Show impact on data shape
    print(f"\nData shape impact:")
    print(f"  Original shape: {features_array.shape}")
    print(f"  New shape: {filtered_features_array.shape}")
    
    return filtered_features_list, filtered_feature_names, removed_features_info

# Function to easily add new files to the configuration

if __name__ == "__main__":
    # Run the main training function
    result = main_xgboost_training()
    
    if result:
        model, scaler, cv_accuracy, segmentation_params = result
        
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Final model trained with cross-validation accuracy: {cv_accuracy:.4f}")
        print("Model and scaler objects are available for further use.")
        print(f"Segmentation used optimized parameters:")
        for key, value in segmentation_params.items():
            print(f"  {key}: {value}")
        print("Model ready for deployment and inference.")
    else:
        print("Training failed. Please check your configuration and data.")