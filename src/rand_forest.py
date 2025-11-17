#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: rand_forest.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Random Forest classifier implementation for RFID tag presence detection.
    Processes RFID tag data to classify dynamic vs static scenarios using
    extracted statistical features with cross-validation and comprehensive
    performance evaluation using optimized arc segmentation parameters.

License: MIT License
Dependencies: sklearn, feature_extraction (local), data_loader (local), arc_segmentation (local)
"""

# filepath: /home/javier/Documents/CFD-rfid-cleaner/src/rand_forest.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

from feature_extraction import extract_features, normalize_features
from data_loader import load_dataset_from_config, print_dataset_summary, validate_dataset

def prepare_arcs_and_labels_rf(tag_data_list, labels, segmentation_params):
    """
    Segment tag data into arcs using advanced arc segmentation for Random Forest training.
    
    :param tag_data_list: List of tag data dictionaries
    :param labels: List of labels for each tag
    :param segmentation_params: Dictionary with segmentation parameters
    :return: Tuple of (arcs, arc_labels)
    """
    from arc_segmentation import segment_tag_data_into_arcs
    
    all_arcs = []
    all_arc_labels = []
    
    print(f"Segmenting {len(tag_data_list)} tags using advanced arc segmentation for Random Forest...")
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

def main_rf_training():
    """
    Main function for Random Forest training with configurable data loading and optimized arc segmentation.
    """
    from config_manager import load_optimized_parameters
    
    print("=== RANDOM FOREST CLASSIFIER FOR RFID TAG ANALYSIS ===\n")
    
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
    
    arcs, arc_labels = prepare_arcs_and_labels_rf(tag_data_list, labels, segmentation_params)
    
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
    
    for i, arc in enumerate(arcs):
        try:
            # Create segment-like dictionary for feature extraction compatibility
            arc_segment = {
                'timestamp': arc['timestamp'],
                'rssi': arc['rssi'],
                'phase': arc['phase']
            }
            
            features = extract_features(arc_segment)
            features_list.append(features)
            valid_labels.append(arc_labels[i])
            
        except Exception as e:
            print(f"  Error extracting features from arc {i+1}: {e}")
            continue
    
    print(f"Features extracted from {len(features_list)} arcs")
    
    if not features_list:
        print("No features extracted. Please check your data and feature extraction.")
        return
    
    # Normalize features
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
    
    # Train Random Forest model with different configurations
    print(f"\n{'='*60}")
    print("MODEL TRAINING PHASE")
    print("="*60)
    
    # Test different Random Forest configurations
    rf_configs = [
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'name': 'RF-100 (default)'},
        {'n_estimators': 500, 'max_depth': None, 'min_samples_split': 2, 'name': 'RF-500'},
        {'n_estimators': 1000, 'max_depth': None, 'min_samples_split': 2, 'name': 'RF-1000'},
        {'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 5, 'name': 'RF-1000 (max_depth=10)'},
        {'n_estimators': 1000, 'max_depth': 20, 'min_samples_split': 10, 'name': 'RF-1000 (max_depth=20)'},
        {'n_estimators': 1500, 'max_depth': None, 'min_samples_split': 2, 'name': 'RF-1500'},
    ]
    
    best_rf = None
    best_cv_score = 0
    best_config = None
    
    print("Testing different Random Forest configurations...")
    
    for config in rf_configs:
        print(f"\n--- {config['name']} ---")
        
        # Create Random Forest classifier
        rf_params = {k: v for k, v in config.items() if k != 'name'}
        clf = RandomForestClassifier(random_state=42, **rf_params)
        
        # 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, features_normalized, valid_labels, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Keep track of best performing model
        if cv_scores.mean() > best_cv_score:
            best_cv_score = cv_scores.mean()
            best_rf = clf
            best_config = config
    
    print(f"\n{'='*60}")
    print("BEST MODEL EVALUATION")
    print("="*60)
    print(f"Best Random Forest: {best_config['name']}")
    print(f"Best CV accuracy: {best_cv_score:.4f}")
    
    # Train best model on training set
    best_rf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Static', 'Dynamic']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation classification report for best model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(best_rf, features_normalized, valid_labels, cv=cv)
    print("\nCross-validation Classification Report:")
    print(classification_report(valid_labels, y_pred_cv, target_names=['Static', 'Dynamic']))
    
    # Feature importance
    print("\n=== FEATURE IMPORTANCE ===")
    feature_importance = best_rf.feature_importances_
    feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
    
    # Sort by importance
    importance_pairs = list(zip(feature_names, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 most important features:")
    for i, (name, importance) in enumerate(importance_pairs[:10]):
        print(f"  {i+1}. {name}: {importance:.4f}")
    
    # Compare all Random Forest configurations summary
    print(f"\n{'='*60}")
    print("ALL CONFIGURATIONS COMPARISON SUMMARY")
    print("="*60)
    
    for config in rf_configs:
        rf_params = {k: v for k, v in config.items() if k != 'name'}
        clf = RandomForestClassifier(random_state=42, **rf_params)
        cv_scores = cross_val_score(clf, features_normalized, valid_labels, cv=cv, scoring='accuracy')
        print(f"{config['name']:<25}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Summary of segmentation performance
    print(f"\n{'='*60}")
    print("SEGMENTATION SUMMARY")
    print("="*60)
    print(f"Original tags: {len(tag_data_list)}")
    print(f"Arcs detected: {len(arcs)}")
    print(f"Arcs per tag (avg): {len(arcs)/len(tag_data_list):.2f}")
    print(f"Segmentation method: Advanced arc segmentation")
    print("Parameters used:")
    for key, value in segmentation_params.items():
        print(f"  {key}: {value}")
    
    return best_rf, scaler, best_cv_score, segmentation_params, best_config

def compare_all_models():
    """
    Compare Random Forest, SVM, and XGBoost performance on the same dataset.
    """
    print("=== RANDOM FOREST vs SVM vs XGBoost COMPARISON ===\n")
    
    results = {}
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_result = main_rf_training()
    
    if rf_result:
        rf_model, rf_scaler, rf_cv_score, rf_seg_params, rf_config = rf_result
        results['Random Forest'] = {
            'model': rf_model,
            'cv_score': rf_cv_score,
            'config': rf_config['name'],
            'details': f"n_estimators={rf_config['n_estimators']}"
        }
    
    # Train SVM
    print("\n" + "="*80)
    print("Training SVM...")
    try:
        from svm_model import main_svm_training
        svm_result = main_svm_training()
        
        if svm_result:
            svm_model, svm_scaler, svm_cv_score, svm_seg_params, svm_config = svm_result
            results['SVM'] = {
                'model': svm_model,
                'cv_score': svm_cv_score,
                'config': svm_config['name'],
                'details': f"kernel={svm_config['kernel']}"
            }
    except Exception as e:
        print(f"SVM training failed: {e}")
    
    # Train XGBoost
    print("\n" + "="*80)
    print("Training XGBoost...")
    try:
        from xgboost_model import main_xgboost_training
        xgb_result = main_xgboost_training()
        
        if xgb_result:
            xgb_model, xgb_scaler, xgb_cv_score, xgb_seg_params = xgb_result
            results['XGBoost'] = {
                'model': xgb_model,
                'cv_score': xgb_cv_score,
                'config': 'XGBoost (GPU)',
                'details': 'n_estimators=1000, max_depth=6'
            }
    except Exception as e:
        print(f"XGBoost training failed: {e}")
    
    # Comparison summary
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("="*80)
    
    if not results:
        print("No models were successfully trained.")
        return
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_score'], reverse=True)
    
    print("Model Performance Ranking:")
    print("-" * 80)
    for rank, (model_name, model_data) in enumerate(sorted_results, 1):
        print(f"{rank}. {model_name}:")
        print(f"   Cross-validation accuracy: {model_data['cv_score']:.4f}")
        print(f"   Configuration: {model_data['config']}")
        print(f"   Details: {model_data['details']}")
        print()
    
    best_model_name = sorted_results[0][0]
    best_score = sorted_results[0][1]['cv_score']
    
    print(f"🏆 Winner: {best_model_name}")
    print(f"🎯 Best CV Accuracy: {best_score:.4f}")
    
    # Performance differences
    if len(sorted_results) > 1:
        print("\nPerformance Gaps:")
        for i in range(1, len(sorted_results)):
            current_name = sorted_results[i][0]
            current_score = sorted_results[i][1]['cv_score']
            gap = best_score - current_score
            print(f"  {best_model_name} vs {current_name}: +{gap:.4f}")
    
    return results

if __name__ == "__main__":
    import sys
    
    print("=== RFID RANDOM FOREST CLASSIFIER SUITE ===\n")
    print("Available modes:")
    print("  1. Random Forest only training")
    print("  2. All models comparison (RF vs SVM vs XGBoost)")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        print("\nRunning ALL MODELS comparison...")
        results = compare_all_models()
    else:
        print("\nRunning Random Forest training...")
        print("(Use 'python rand_forest.py compare' for all models comparison)")
        
        result = main_rf_training()
        
        if result:
            model, scaler, cv_accuracy, segmentation_params, config = result
            
            print(f"\n{'='*60}")
            print("TRAINING SUMMARY")
            print("="*60)
            print(f"Best Random Forest model: {config['name']}")
            print(f"Cross-validation accuracy: {cv_accuracy:.4f}")
            print("Model and scaler objects are available for further use.")
            print("Segmentation used optimized parameters:")
            for key, value in segmentation_params.items():
                print(f"  {key}: {value}")
            print("Model ready for deployment and inference.")
        else:
            print("Training failed. Please check your configuration and data.")