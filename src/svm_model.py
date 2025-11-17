#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: svm_model.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Support Vector Machine (SVM) implementation for RFID tag presence detection.
    Processes RFID tag data to classify dynamic vs static scenarios using
    extracted statistical features with cross-validation and comprehensive
    performance evaluation using optimized arc segmentation parameters.

License: MIT License
Dependencies: sklearn, feature_extraction (local), data_loader (local), arc_segmentation (local)
"""

from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

from feature_extraction import extract_features, normalize_features
from data_loader import load_dataset_from_config, print_dataset_summary, validate_dataset

def prepare_arcs_and_labels_svm(tag_data_list, labels, segmentation_params):
    """
    Segment tag data into arcs using advanced arc segmentation for SVM training.
    
    :param tag_data_list: List of tag data dictionaries
    :param labels: List of labels for each tag
    :param segmentation_params: Dictionary with segmentation parameters
    :return: Tuple of (arcs, arc_labels)
    """
    from arc_segmentation import segment_tag_data_into_arcs
    
    all_arcs = []
    all_arc_labels = []
    
    print(f"Segmenting {len(tag_data_list)} tags using advanced arc segmentation for SVM...")
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

def main_svm_training():
    """
    Main function for SVM training with configurable data loading and optimized arc segmentation.
    """
    from config_manager import load_optimized_parameters
    
    print("=== SVM CLASSIFIER FOR RFID TAG ANALYSIS ===\n")
    
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
        optimized_params = load_optimized_parameters('output_data/optimized_parameters.json')
        
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
    
    arcs, arc_labels = prepare_arcs_and_labels_svm(tag_data_list, labels, segmentation_params)
    
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
    
    if feature_names_descriptive is None:
        print("No feature names detected. Check feature extraction.")
        return
    
    # Feature selection/removal phase
    print(f"\n{'='*60}")
    print("FEATURE SELECTION PHASE")
    print("="*60)
    
    # Specify which features to remove for SVM testing
    features_to_remove = [] # ['time_diff_min', 'time_diff_max', 'total_time', 'num_samples', 'time_diff_mean', 'time_diff_std']  # Ejemplo: remover features temporales que pueden tener poca variabilidad
    # features_to_remove = ['rssi_min', 'rssi_max']  # Ejemplo: remover extremos de RSSI
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
    
    # Normalize features (now with filtered features)
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
    
    # Train SVM model with different kernels
    print(f"\n{'='*60}")
    print("MODEL TRAINING PHASE")
    print("="*60)
    
    # Test different SVM configurations
    svm_configs = [
        {'kernel': 'linear', 'C': 1.0, 'name': 'Linear SVM'},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'name': 'RBF SVM'},
        {'kernel': 'poly', 'degree': 3, 'C': 1.0, 'name': 'Polynomial SVM (degree=3)'},
        {'kernel': 'poly', 'degree': 5, 'C': 1.0, 'name': 'Polynomial SVM (degree=5)'},
        {'kernel': 'poly', 'degree': 10, 'C': 1.0, 'name': 'Polynomial SVM (degree=10)'}
    ]
    
    best_svm = None
    best_cv_score = 0
    best_config = None
    
    print("Testing different SVM configurations...")
    
    for config in svm_configs:
        print(f"\n--- {config['name']} ---")
        
        # Create SVM classifier
        svm_params = {k: v for k, v in config.items() if k != 'name'}
        clf = svm.SVC(**svm_params)
        
        # 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, features_normalized, valid_labels, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Keep track of best performing model
        if cv_scores.mean() > best_cv_score:
            best_cv_score = cv_scores.mean()
            best_svm = clf
            best_config = config
    
    print(f"\n{'='*60}")
    print("BEST MODEL EVALUATION")
    print("="*60)
    print(f"Best SVM: {best_config['name']}")
    print(f"Best CV accuracy: {best_cv_score:.4f}")
    
    # Train best model on training set
    best_svm.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Static', 'Dynamic']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation classification report for best model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(best_svm, features_normalized, valid_labels, cv=cv)
    print("\nCross-validation Classification Report:")
    print(classification_report(valid_labels, y_pred_cv, target_names=['Static', 'Dynamic']))
    
    # Feature analysis (disponible para TODOS los kernels ahora)
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    if best_config['kernel'] == 'linear':
        print("Using LINEAR SVM coefficients (direct weights):")
        print("-" * 50)
        
        # Get feature coefficients (weights) for linear SVM
        feature_weights = abs(best_svm.coef_[0])
        
        # Create pairs of (feature_name, weight)
        weight_pairs = list(zip(filtered_feature_names, feature_weights))
        weight_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, weight) in enumerate(weight_pairs):
            print(f"  {i+1:2d}. {name:<20}: {weight:.4f}")
        
        # Store for metadata
        importance_method = "linear_weights"
        importance_values = dict(weight_pairs)
        
    else:
        print(f"Using PERMUTATION IMPORTANCE for {best_config['kernel']} kernel:")
        print("-" * 50)
        print("(Measuring accuracy drop when each feature is randomly shuffled)")
        
        # Use permutation importance for non-linear kernels
        from sklearn.inspection import permutation_importance
        
        perm_importance = permutation_importance(
            best_svm, features_normalized, valid_labels, 
            n_repeats=5, 
            random_state=42, 
            scoring='accuracy'
        )
        
        # Create pairs and sort
        importance_pairs = list(zip(filtered_feature_names, perm_importance.importances_mean))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, importance) in enumerate(importance_pairs):
            std = perm_importance.importances_std[filtered_feature_names.index(name)]
            print(f"  {i+1:2d}. {name:<20}: {importance:.4f} ± {std:.4f}")
        
        # Store for metadata
        importance_method = "permutation"
        importance_values = dict(importance_pairs)

    # Feature analysis by category (works for both methods)
    print(f"\nFeature importance by category ({importance_method}):")
    print("-" * 50)

    # Group by prefix
    if importance_method == "linear_weights":
        analysis_pairs = weight_pairs
    else:
        analysis_pairs = importance_pairs

    rssi_features = [pair for pair in analysis_pairs if pair[0].startswith('rssi_')]
    phase_features = [pair for pair in analysis_pairs if pair[0].startswith('phase_')]
    time_features = [pair for pair in analysis_pairs if any(keyword in pair[0] for keyword in ['time_', 'total_time', 'num_samples'])]

    if rssi_features:
        rssi_total = sum(weight for _, weight in rssi_features)
        rssi_avg = rssi_total / len(rssi_features)
        print(f"  RSSI features ({len(rssi_features)} total):")
        print(f"    Total weight: {rssi_total:.4f}")
        print(f"    Average weight: {rssi_avg:.4f}")
        print(f"    Highest weight: {rssi_features[0][0]} ({rssi_features[0][1]:.4f})")
        
        print(f"    Individual RSSI features:")
        for name, weight in rssi_features:
            print(f"      - {name}: {weight:.4f}")
    
    if phase_features:
        phase_total = sum(weight for _, weight in phase_features)
        phase_avg = phase_total / len(phase_features)
        print(f"\n  Phase features ({len(phase_features)} total):")
        print(f"    Total weight: {phase_total:.4f}")
        print(f"    Average weight: {phase_avg:.4f}")
        print(f"    Highest weight: {phase_features[0][0]} ({phase_features[0][1]:.4f})")
        
        print(f"    Individual Phase features:")
        for name, weight in phase_features:
            print(f"      - {name}: {weight:.4f}")
    
    if time_features:
        time_total = sum(weight for _, weight in time_features)
        time_avg = time_total / len(time_features)
        print(f"\n  Temporal features ({len(time_features)} total):")
        print(f"    Total weight: {time_total:.4f}")
        print(f"    Average weight: {time_avg:.4f}")
        print(f"    Highest weight: {time_features[0][0]} ({time_features[0][1]:.4f})")
        
        print(f"    Individual Temporal features:")
        for name, weight in time_features:
            print(f"      - {name}: {weight:.4f}")
    else:
        print(f"\nNote: Feature importance analysis not available for {best_config['kernel']} kernel.")
        print("Only linear SVM provides interpretable feature weights.")
    
    # Compare all SVM kernels summary
    print(f"\n{'='*60}")
    print("ALL KERNELS COMPARISON SUMMARY")
    print("="*60)
    
    for config in svm_configs:
        svm_params = {k: v for k, v in config.items() if k != 'name'}
        clf = svm.SVC(**svm_params)
        cv_scores = cross_val_score(clf, features_normalized, valid_labels, cv=cv, scoring='accuracy')
        print(f"{config['name']:<25}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Save model and metadata
    print(f"\n{'='*60}")
    print("SAVING MODEL AND METADATA")
    print("="*60)
    
    import os
    import json
    import joblib
    os.makedirs('output_data', exist_ok=True)
    
    # Save SVM model
    joblib.dump(best_svm, 'output_data/svm_model.pkl')
    
    # Save scaler
    joblib.dump(scaler, 'output_data/svm_scaler.pkl')
    
    # Save model metadata
    svm_metadata = {
        'model_type': 'SVM',
        'best_kernel': best_config['kernel'],
        'best_config': best_config,
        'feature_names': filtered_feature_names,
        'features_removed': features_to_remove,
        'removal_info': removal_info,
        'model_performance': {
            'cv_accuracy_mean': float(best_cv_score),
            'test_accuracy': float(test_accuracy)
        },
        'segmentation_params': segmentation_params
    }
    
    # Add feature weights if linear SVM
    if best_config['kernel'] == 'linear':
        svm_metadata['feature_weights'] = {
            name: float(weight) 
            for name, weight in zip(filtered_feature_names, abs(best_svm.coef_[0]))
        }
        svm_metadata['feature_weights_sorted'] = [
            {'name': name, 'weight': float(weight)} 
            for name, weight in weight_pairs
        ]
    
    with open('output_data/svm_model_metadata.json', 'w') as f:
        json.dump(svm_metadata, f, indent=2)
    
    print("✓ Model saved to: output_data/svm_model.pkl")
    print("✓ Scaler saved to: output_data/svm_scaler.pkl")
    print("✓ Model metadata saved to: output_data/svm_model_metadata.json")
    
    # Summary of segmentation performance
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Original tags: {len(tag_data_list)}")
    print(f"Arcs detected: {len(arcs)}")
    print(f"Valid arcs processed: {len(valid_labels)}")
    print(f"Arcs per tag (avg): {len(arcs)/len(tag_data_list):.2f}")
    print(f"Features extracted per arc: {len(filtered_feature_names)}")
    print(f"Cross-validation accuracy: {best_cv_score:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Best SVM kernel: {best_config['kernel']}")
    
    if features_to_remove:
        print(f"\nFeatures removed for this run: {features_to_remove}")
    
    print(f"\nSegmentation method: Advanced arc segmentation")
    print("Parameters used:")
    for key, value in segmentation_params.items():
        print(f"  {key}: {value}")
    
    return best_svm, scaler, best_cv_score, segmentation_params, best_config

def calculate_permutation_importance(model, X, y, feature_names, n_repeats=10):
    """
    Calculate feature importance by permuting each feature and measuring performance drop.
    Works with ANY kernel!
    """
    from sklearn.inspection import permutation_importance
    
    print(f"\n{'='*60}")
    print("PERMUTATION FEATURE IMPORTANCE (WORKS WITH ANY KERNEL)")
    print("="*60)
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X, y, 
        n_repeats=n_repeats, 
        random_state=42, 
        scoring='accuracy'
    )
    
    # Create pairs of (feature_name, importance)
    importance_pairs = list(zip(feature_names, perm_importance.importances_mean))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("Feature importance (accuracy drop when permuted):")
    print("-" * 60)
    for i, (name, importance) in enumerate(importance_pairs):
        std = perm_importance.importances_std[feature_names.index(name)]
        print(f"  {i+1:2d}. {name:<20}: {importance:.4f} ± {std:.4f}")
    
    return importance_pairs, perm_importance.importances_std

def calculate_shap_importance(model, X_train, X_test, feature_names):
    """
    Calculate SHAP values for feature importance. Works with any kernel!
    """
    try:
        import shap
        
        print(f"\n{'='*60}")
        print("SHAP FEATURE IMPORTANCE (WORKS WITH ANY KERNEL)")
        print("="*60)
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(model.predict, X_train[:100])  # Sample for speed
        shap_values = explainer.shap_values(X_test[:50])  # Sample for speed
        
        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create pairs and sort
        shap_pairs = list(zip(feature_names, mean_shap))
        shap_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print("Feature importance (mean |SHAP| values):")
        print("-" * 50)
        for i, (name, importance) in enumerate(shap_pairs):
            print(f"  {i+1:2d}. {name:<20}: {importance:.4f}")
        
        return shap_pairs
        
    except ImportError:
        print("SHAP not installed. Use: pip install shap")
        return None


def analyze_feature_importance_rfe(model, X, y, feature_names):
    """
    Use Recursive Feature Elimination to rank features. Works with any kernel!
    """
    from sklearn.feature_selection import RFE
    
    print(f"\n{'='*60}")
    print("RECURSIVE FEATURE ELIMINATION (WORKS WITH ANY KERNEL)")
    print("="*60)
    
    # RFE to rank all features
    rfe = RFE(model, n_features_to_select=1)
    rfe.fit(X, y)
    
    # Get rankings (1 = most important)
    rankings = rfe.ranking_
    
    # Create pairs and sort by ranking
    rank_pairs = list(zip(feature_names, rankings))
    rank_pairs.sort(key=lambda x: x[1])  # Sort by ranking (lower = better)
    
    print("Feature importance ranking (1 = most important):")
    print("-" * 50)
    for i, (name, rank) in enumerate(rank_pairs):
        print(f"  Rank {rank:2d}. {name:<20}")
    
    return rank_pairs

def compare_svm_xgboost():
    """
    Compare SVM and XGBoost performance on the same dataset.
    """
    print("=== SVM vs XGBoost COMPARISON ===\n")
    
    # Train SVM
    print("Training SVM...")
    svm_result = main_svm_training()
    
    if not svm_result:
        print("SVM training failed.")
        return
    
    svm_model, svm_scaler, svm_cv_score, svm_seg_params, svm_config = svm_result
    
    # Train XGBoost
    print("\n" + "="*80)
    print("Training XGBoost...")
    from xgboost_model import main_xgboost_training
    
    xgb_result = main_xgboost_training()
    
    if not xgb_result:
        print("XGBoost training failed.")
        return
    
    xgb_model, xgb_scaler, xgb_cv_score, xgb_seg_params = xgb_result
    
    # Comparison summary
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    print(f"SVM ({svm_config['name']}):")
    print(f"  Cross-validation accuracy: {svm_cv_score:.4f}")
    print(f"  Kernel: {svm_config['kernel']}")
    
    print(f"\nXGBoost:")
    print(f"  Cross-validation accuracy: {xgb_cv_score:.4f}")
    print(f"  Trees: 1000, Max depth: 6")
    
    winner = "SVM" if svm_cv_score > xgb_cv_score else "XGBoost"
    difference = abs(svm_cv_score - xgb_cv_score)
    
    print(f"\nWinner: {winner}")
    print(f"Performance difference: {difference:.4f}")
    
    return svm_result, xgb_result

if __name__ == "__main__":
    import sys
    
    print("=== RFID SVM CLASSIFIER SUITE ===\n")
    print("Available modes:")
    print("  1. SVM only training")
    print("  2. SVM vs XGBoost comparison")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        print("\nRunning SVM vs XGBoost comparison...")
        svm_result, xgb_result = compare_svm_xgboost()
    else:
        print("\nRunning SVM training...")
        print("(Use 'python svm_model.py compare' for SVM vs XGBoost comparison)")
        
        result = main_svm_training()
        
        if result:
            model, scaler, cv_accuracy, segmentation_params, config = result
            
            print(f"\n{'='*60}")
            print("TRAINING SUMMARY")
            print("="*60)
            print(f"Best SVM model: {config['name']}")
            print(f"Cross-validation accuracy: {cv_accuracy:.4f}")
            print("Model and scaler objects are available for further use.")
            print("Segmentation used optimized parameters:")
            for key, value in segmentation_params.items():
                print(f"  {key}: {value}")
            print("Model ready for deployment and inference.")
        else:
            print("Training failed. Please check your configuration and data.")