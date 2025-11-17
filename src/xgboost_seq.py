#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: xgboost_seq.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    XGBoost classifier implementation for RFID tag presence detection with interpolation-based features.
    Processes RFID tag data to classify dynamic vs static scenarios using
    interpolated RSSI and phase values from arc segmentation with cross-validation 
    and comprehensive performance evaluation utilizing CUDA for enhanced performance.

License: MIT License
Dependencies: xgboost, sklearn, cupy, interpolation (local), data_loader (local), arc_segmentation (local)
"""

from xgboost import XGBClassifier, DMatrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import cupy
import numpy as np
import joblib

from interpolation import interpolate_and_smooth_segment
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

def prepare_interpolated_features_from_arcs(arcs, arc_labels, num_points=100):
    """
    Interpolate each arc to a fixed number of points and prepare features for XGBoost.
    
    :param arcs: List of arc dictionaries from arc segmentation
    :param arc_labels: List of labels for each arc
    :param num_points: Number of points to interpolate each arc to
    :return: Tuple of (features_array, valid_labels)
    """
    print(f"Preparing interpolated features from {len(arcs)} arcs...")
    print(f"Interpolating each arc to {num_points} equidistant time points...")
    
    features_list = []
    valid_labels = []
    
    for i, (arc, label) in enumerate(zip(arcs, arc_labels)):
        try:
            # Skip arcs that are too short for meaningful interpolation
            if len(arc['timestamp']) < 3:
                print(f"  Skipping arc {i+1}: too few samples ({len(arc['timestamp'])})")
                continue
            
            # Use interpolation module to interpolate arc to fixed number of points
            interpolated = interpolate_and_smooth_segment(
                arc,
                num_points=num_points,
                kind='linear',
                smoothing_method='gaussian',
                smoothing_params={'sigma': 1.0}
            )
            
            if interpolated is None or 'rssi' not in interpolated or 'phase' not in interpolated:
                print(f"  Skipping arc {i+1}: interpolation failed")
                continue
            
            # Verify interpolation produced correct number of points
            if len(interpolated['rssi']) != num_points or len(interpolated['phase']) != num_points:
                print(f"  Skipping arc {i+1}: interpolation produced wrong number of points")
                continue
            
            # Create feature vector: concatenate RSSI and phase values
            # Feature vector will have shape (2 * num_points,)
            feature_vector = np.concatenate([
                interpolated['rssi'],    # First num_points features
                interpolated['phase']    # Next num_points features
            ])
            
            features_list.append(feature_vector)
            valid_labels.append(label)
            
            # Print progress for first few arcs
            if i < 5:
                rssi_range = f"{np.min(interpolated['rssi']):.2f} to {np.max(interpolated['rssi']):.2f}"
                phase_range = f"{np.min(interpolated['phase']):.1f}° to {np.max(interpolated['phase']):.1f}°"
                print(f"    Arc {i+1}: {len(arc['timestamp'])} → {num_points} points, "
                      f"RSSI: {rssi_range} dBm, Phase: {phase_range}")
            
        except Exception as e:
            print(f"  Error processing arc {i+1}: {e}")
            continue
    
    if not features_list:
        print("No valid features extracted from arcs.")
        return np.array([]), []
    
    # Convert to numpy array
    features_array = np.array(features_list)
    
    print(f"Successfully processed {len(features_list)} arcs")
    print(f"Feature matrix shape: {features_array.shape}")
    print(f"Features per arc: {features_array.shape[1]} ({num_points} RSSI + {num_points} Phase)")
    
    # Print feature statistics
    print(f"Feature statistics:")
    rssi_features = features_array[:, :num_points]
    phase_features = features_array[:, num_points:]
    
    print(f"  RSSI range: {np.min(rssi_features):.2f} to {np.max(rssi_features):.2f} dBm")
    print(f"  Phase range: {np.min(phase_features):.1f}° to {np.max(phase_features):.1f}°")
    print(f"  RSSI mean: {np.mean(rssi_features):.2f} ± {np.std(rssi_features):.2f} dBm")
    print(f"  Phase mean: {np.mean(phase_features):.1f}° ± {np.std(phase_features):.1f}°")
    
    return features_array, valid_labels

def main_xgboost_training():
    """
    Main function for XGBoost training with configurable data loading and interpolation-based features.
    """
    from config_manager import load_optimized_parameters
    
    print("=== XGBOOST CLASSIFIER FOR RFID TAG ANALYSIS (INTERPOLATION-BASED) ===\n")
    
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
    
    # Prepare interpolated features from arcs
    print(f"\n{'='*60}")
    print("INTERPOLATION-BASED FEATURE PREPARATION")
    print("="*60)
    
    # Use 100 interpolation points for each arc
    interpolation_points = 100
    features_array, valid_labels = prepare_interpolated_features_from_arcs(
        arcs, arc_labels, num_points=interpolation_points
    )
    
    if len(features_array) == 0:
        print("No valid features extracted. Please check your data and interpolation parameters.")
        return
    
    # Normalize features
    print(f"\n{'='*60}")
    print("FEATURE NORMALIZATION")
    print("="*60)
    print("Normalizing interpolated features...")
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_array)
    
    print(f"Feature normalization completed:")
    print(f"  Original feature range: [{np.min(features_array):.3f}, {np.max(features_array):.3f}]")
    print(f"  Normalized feature range: [{np.min(features_normalized):.3f}, {np.max(features_normalized):.3f}]")
    
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
    print("Training XGBoost model with interpolated features...")
    print(f"Input features: {X_train.shape[1]} ({interpolation_points} RSSI + {interpolation_points} Phase)")
    
    clf = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist',
        device='cuda',
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    clf.fit(X_train, y_train)
    print("Training completed!")
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print("="*60)
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Static', 'Dynamic']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 5-fold cross-validation
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION")
    print("="*60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, features_normalized, valid_labels, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Classification report for cross-validation
    y_pred_cv = cross_val_predict(clf, features_normalized, valid_labels, cv=cv)
    print("\nCross-validation Classification Report:")
    print(classification_report(valid_labels, y_pred_cv, target_names=['Static', 'Dynamic']))
    
    # Feature importance analysis
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    feature_importance = clf.feature_importances_
    
    # Separate RSSI and Phase feature importances
    rssi_importance = feature_importance[:interpolation_points]
    phase_importance = feature_importance[interpolation_points:]
    
    print(f"Feature importance statistics:")
    print(f"  RSSI features - Mean: {np.mean(rssi_importance):.4f}, Max: {np.max(rssi_importance):.4f}")
    print(f"  Phase features - Mean: {np.mean(phase_importance):.4f}, Max: {np.max(phase_importance):.4f}")
    
    # Find most important features
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    print(f"\nTop 10 most important features:")
    for i, idx in enumerate(top_indices):
        feature_type = "RSSI" if idx < interpolation_points else "Phase"
        position = idx if idx < interpolation_points else idx - interpolation_points
        print(f"  {i+1:2d}. {feature_type} point {position:3d}: {feature_importance[idx]:.4f}")
    
    # Save model and scaler
    print(f"\n{'='*60}")
    print("SAVING MODEL AND SCALER")
    print("="*60)
    
    import os
    os.makedirs('output_data', exist_ok=True)
    
    # Save XGBoost model
    clf.save_model('output_data/xgboost_interpolation_model.json')
    
    # Save scaler
    joblib.dump(scaler, 'output_data/xgboost_interpolation_scaler.pkl')
    
    # Save training configuration
    training_config = {
        'model_type': 'XGBoost_Interpolation',
        'interpolation_points': interpolation_points,
        'feature_count': X_train.shape[1],
        'segmentation_params': segmentation_params,
        'cv_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'test_accuracy': float(test_accuracy),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'total_arcs': len(arcs),
        'valid_arcs': len(valid_labels)
    }
    
    import json
    with open('output_data/xgboost_interpolation_config.json', 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print("✓ Model saved to: output_data/xgboost_interpolation_model.json")
    print("✓ Scaler saved to: output_data/xgboost_interpolation_scaler.pkl")
    print("✓ Configuration saved to: output_data/xgboost_interpolation_config.json")
    
    # Summary of segmentation and interpolation performance
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Segmentation method: Advanced arc segmentation")
    print(f"Feature method: Interpolation-based (RSSI + Phase)")
    print(f"Original tags: {len(tag_data_list)}")
    print(f"Total arcs detected: {len(arcs)}")
    print(f"Valid arcs processed: {len(valid_labels)}")
    print(f"Arcs per tag (avg): {len(arcs)/len(tag_data_list):.2f}")
    print(f"Interpolation points per arc: {interpolation_points}")
    print(f"Total features per arc: {interpolation_points * 2} (RSSI + Phase)")
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    print("\nSegmentation parameters used:")
    for key, value in segmentation_params.items():
        print(f"  {key}: {value}")
    
    return clf, scaler, cv_scores.mean(), segmentation_params

if __name__ == "__main__":
    # Run the main training function
    result = main_xgboost_training()
    
    if result:
        model, scaler, cv_accuracy, segmentation_params = result
        
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print("="*60)
        print(f"XGBoost model trained successfully with interpolation-based features!")
        print(f"Final cross-validation accuracy: {cv_accuracy:.4f}")
        print("Model, scaler, and configuration saved for future use.")
        print("Model ready for deployment and inference.")
    else:
        print("Training failed. Please check your configuration and data.")