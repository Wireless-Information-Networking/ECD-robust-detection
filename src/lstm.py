#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: lstm.py
Author: Javier del Río
Date: 2025-10-22
Description: 
    PyTorch LSTM neural network implementation for RFID tag presence detection.
    Processes RFID tag data to classify dynamic vs static scenarios using
    arc segmentation results directly as sequential features with optimized
    segmentation parameters and comprehensive performance evaluation.

License: MIT License
Dependencies: torch, sklearn, arc_segmentation (local), data_loader (local)
"""

# filepath: /home/javier/Documents/CFD-rfid-cleaner/src/lstm.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import json
import os

from data_loader import load_dataset_from_config, print_dataset_summary, validate_dataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LSTMClassifier(nn.Module):
    """
    PyTorch LSTM model for binary classification of RFID tag data.
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout_rate=0.3):
        """
        Initialize LSTM classifier.
        
        :param input_size: Number of features per timestep
        :param hidden_size: Hidden layer size
        :param num_layers: Number of LSTM layers
        :param dropout_rate: Dropout rate for regularization
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = nn.Linear(50, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass.
        
        :param x: Input tensor of shape (batch_size, seq_length, input_size)
        :return: Output tensor of shape (batch_size, 1)
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Batch normalization
        last_output = self.batch_norm(last_output)
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        
        return out

def prepare_arcs_and_labels_lstm(tag_data_list, labels, segmentation_params):
    """
    Segment tag data into arcs using advanced arc segmentation for LSTM training.
    
    :param tag_data_list: List of tag data dictionaries
    :param labels: List of labels for each tag
    :param segmentation_params: Dictionary with segmentation parameters
    :return: Tuple of (arcs, arc_labels)
    """
    from arc_segmentation import segment_tag_data_into_arcs
    
    all_arcs = []
    all_arc_labels = []
    
    print(f"Segmenting {len(tag_data_list)} tags using advanced arc segmentation for PyTorch LSTM...")
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

def prepare_sequences_from_arcs(arcs, sequence_length=100):
    """
    Prepare sequential data from arcs for PyTorch LSTM training.
    Each arc becomes a sequence with RSSI and phase values interpolated to fixed length.
    
    :param arcs: List of arc dictionaries from arc segmentation
    :param sequence_length: Fixed length for all sequences (interpolating)
    :return: Array of sequences shaped (n_arcs, sequence_length, n_features)
    """
    from scipy.interpolate import interp1d
    
    print(f"Preparing sequences from {len(arcs)} arcs with interpolated sequence length {sequence_length}...")
    
    sequences = []
    valid_arc_indices = []
    
    for i, arc in enumerate(arcs):
        try:
            # Extract time series data from arc
            rssi_values = arc['rssi']
            phase_values = arc['phase']
            timestamps = arc['timestamp']
            
            # Skip arcs that are too short for meaningful interpolation
            if len(rssi_values) < 3:
                print(f"  Skipping arc {i+1}: too few samples ({len(rssi_values)})")
                continue
            
            # Normalize time to [0, 1] for this arc
            time_normalized = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else np.zeros_like(timestamps)
            
            # Create new time grid for interpolation (fixed points)
            new_time_grid = np.linspace(0, 1, sequence_length)
            
            # Interpolate RSSI values
            if len(rssi_values) == 1:
                # If only one point, replicate it
                rssi_interpolated = np.full(sequence_length, rssi_values[0])
            else:
                # Use linear interpolation with extrapolation for edge cases
                rssi_interp_func = interp1d(
                    time_normalized, rssi_values, 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=(rssi_values[0], rssi_values[-1])
                )
                rssi_interpolated = rssi_interp_func(new_time_grid)
                
                # Handle any remaining NaN values
                rssi_interpolated = np.nan_to_num(rssi_interpolated, 
                                                nan=np.mean(rssi_values),
                                                posinf=np.max(rssi_values),
                                                neginf=np.min(rssi_values))
            
            # Interpolate Phase values (handling wraparound)
            if len(phase_values) == 1:
                # If only one point, replicate it
                phase_interpolated = np.full(sequence_length, phase_values[0])
            else:
                # Unwrap phase to handle 0/360 degree transitions
                phase_unwrapped = np.unwrap(np.radians(phase_values))
                
                # Interpolate unwrapped phase
                phase_interp_func = interp1d(
                    time_normalized, phase_unwrapped,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(phase_unwrapped[0], phase_unwrapped[-1])
                )
                phase_interpolated_unwrapped = phase_interp_func(new_time_grid)
                
                # Convert back to degrees and wrap to [0, 360)
                phase_interpolated = np.degrees(phase_interpolated_unwrapped) % 360
                
                # Handle any remaining NaN values
                phase_interpolated = np.nan_to_num(phase_interpolated,
                                                 nan=np.mean(phase_values),
                                                 posinf=np.max(phase_values),
                                                 neginf=np.min(phase_values))
            
            # Create feature matrix: [rssi, phase, normalized_time]
            features = np.column_stack([
                rssi_interpolated,
                phase_interpolated, 
                new_time_grid
            ])
            
            # Verify the shape is correct
            if features.shape != (sequence_length, 3):
                print(f"  Warning: Arc {i+1} has unexpected shape {features.shape}, skipping...")
                continue
            
            # Check for any invalid values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                print(f"  Warning: Arc {i+1} contains invalid values after interpolation, skipping...")
                continue
            
            sequences.append(features)
            valid_arc_indices.append(i)
            
            # Print detailed info for first few arcs
            if i < 5:
                original_length = len(rssi_values)
                rssi_range = f"{np.min(rssi_interpolated):.2f} to {np.max(rssi_interpolated):.2f}"
                phase_range = f"{np.min(phase_interpolated):.1f}° to {np.max(phase_interpolated):.1f}°"
                print(f"    Arc {i+1}: {original_length} → {sequence_length} points, "
                      f"RSSI: {rssi_range} dBm, Phase: {phase_range}")
            
        except Exception as e:
            print(f"  Error processing arc {i+1}: {e}")
            continue
    
    sequences = np.array(sequences)
    print(f"Created {len(sequences)} interpolated sequences with shape {sequences.shape}")
    
    # Print summary statistics
    if len(sequences) > 0:
        print(f"Sequence statistics:")
        print(f"  RSSI range: {np.min(sequences[:, :, 0]):.2f} to {np.max(sequences[:, :, 0]):.2f} dBm")
        print(f"  Phase range: {np.min(sequences[:, :, 1]):.1f}° to {np.max(sequences[:, :, 1]):.1f}°")
        print(f"  Time range: {np.min(sequences[:, :, 2]):.3f} to {np.max(sequences[:, :, 2]):.3f}")
    
    return sequences, valid_arc_indices

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, patience=20):
    """
    Train PyTorch LSTM model with early stopping.
    
    :param model: PyTorch model
    :param train_loader: Training data loader
    :param val_loader: Validation data loader
    :param num_epochs: Maximum number of epochs
    :param learning_rate: Learning rate
    :param patience: Early stopping patience
    :return: Training history
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training model on {device}...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y.float()).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.float())
                
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y.float()).sum().item()
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or patience_counter >= patience:
            print(f'Epoch [{epoch+1}/{num_epochs}] - '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies,
        'epochs_trained': len(train_losses)
    }
    
    return history

def evaluate_model(model, test_loader):
    """
    Evaluate PyTorch model on test set.
    
    :param model: Trained PyTorch model
    :param test_loader: Test data loader
    :return: Predictions and true labels
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            predictions = (outputs.squeeze() > 0.5).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(batch_y.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

def main_lstm_training():
    """
    Main function for PyTorch LSTM training with configurable data loading and optimized arc segmentation.
    """
    from config_manager import load_optimized_parameters
    
    print("=== PYTORCH LSTM CLASSIFIER FOR RFID TAG ANALYSIS ===\n")
    
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
    
    arcs, arc_labels = prepare_arcs_and_labels_lstm(tag_data_list, labels, segmentation_params)
    
    if not arcs:
        print("No arcs created. Please check segmentation parameters.")
        return
    
    # Prepare sequences from arcs
    print(f"\n{'='*60}")
    print("SEQUENCE PREPARATION PHASE")
    print("="*60)
    
    sequence_length = 50  # Fixed sequence length for LSTM
    sequences, valid_indices = prepare_sequences_from_arcs(arcs, sequence_length)
    
    if len(sequences) == 0:
        print("No valid sequences created. Please check your data.")
        return
    
    # Filter labels to match valid sequences
    valid_labels = [arc_labels[i] for i in valid_indices]
    
    print(f"Final dataset: {len(sequences)} sequences with shape {sequences.shape}")
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    # Normalize features across all sequences
    print("\nNormalizing sequences...")
    n_samples, seq_len, n_features = sequences.shape
    
    # Reshape for normalization
    sequences_reshaped = sequences.reshape(-1, n_features)
    
    # Fit scaler and transform
    scaler = StandardScaler()
    sequences_normalized = scaler.fit_transform(sequences_reshaped)
    
    # Reshape back to sequences
    sequences_normalized = sequences_normalized.reshape(n_samples, seq_len, n_features)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        sequences_normalized, valid_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=valid_labels
    )
    
    print(f"Training set: {len(X_train)} sequences")
    print(f"Test set: {len(X_test)} sequences")
    
    # Further split training set for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_split)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train_split)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and train model
    print(f"\n{'='*60}")
    print("MODEL TRAINING PHASE")
    print("="*60)
    
    # Initialize model
    input_size = n_features
    model = LSTMClassifier(input_size=input_size, hidden_size=128, num_layers=2, dropout_rate=0.3)
    model = model.to(device)
    
    print("PyTorch LSTM Model Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    history = train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, patience=20)
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print("="*60)
    
    y_pred, y_true = evaluate_model(model, test_loader)
    test_accuracy = accuracy_score(y_true, y_pred)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Static', 'Dynamic']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Cross-validation
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION")
    print("="*60)
    
    cv_scores = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(sequences_normalized, valid_labels)):
        print(f"\nFold {fold + 1}/5...")
        
        X_train_cv = torch.FloatTensor(sequences_normalized[train_idx])
        X_val_cv = torch.FloatTensor(sequences_normalized[val_idx])
        y_train_cv = torch.LongTensor([valid_labels[i] for i in train_idx])
        y_val_cv = torch.LongTensor([valid_labels[i] for i in val_idx])
        
        # Create data loaders for this fold
        train_cv_dataset = TensorDataset(X_train_cv, y_train_cv)
        val_cv_dataset = TensorDataset(X_val_cv, y_val_cv)
        train_cv_loader = DataLoader(train_cv_dataset, batch_size=batch_size, shuffle=True)
        val_cv_loader = DataLoader(val_cv_dataset, batch_size=batch_size, shuffle=False)
        
        # Create fresh model for this fold
        model_cv = LSTMClassifier(input_size=input_size, hidden_size=128, num_layers=2, dropout_rate=0.3)
        model_cv = model_cv.to(device)
        
        # Train with reduced epochs for CV
        _ = train_model(model_cv, train_cv_loader, val_cv_loader, num_epochs=50, patience=15)
        
        # Evaluate
        y_pred_cv, y_true_cv = evaluate_model(model_cv, val_cv_loader)
        fold_accuracy = accuracy_score(y_true_cv, y_pred_cv)
        cv_scores.append(fold_accuracy)
        
        print(f"  Fold {fold + 1} accuracy: {fold_accuracy:.4f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Save model and scaler
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print("="*60)
    
    os.makedirs('output_data', exist_ok=True)
    
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout_rate': 0.3,
            'sequence_length': sequence_length
        },
        'training_config': segmentation_params
    }, 'output_data/pytorch_lstm_model.pth')
    
    # Save scaler
    joblib.dump(scaler, 'output_data/pytorch_lstm_scaler.pkl')
    
    # Save training history
    with open('output_data/pytorch_lstm_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("✓ Model saved to: output_data/pytorch_lstm_model.pth")
    print("✓ Scaler saved to: output_data/pytorch_lstm_scaler.pkl")
    print("✓ Training history saved to: output_data/pytorch_lstm_training_history.json")
    
    # Summary of segmentation performance
    print(f"\n{'='*60}")
    print("SEGMENTATION SUMMARY")
    print("="*60)
    print(f"Original tags: {len(tag_data_list)}")
    print(f"Arcs detected: {len(arcs)}")
    print(f"Valid sequences: {len(sequences)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Features per timestep: {n_features}")
    print(f"Device used: {device}")
    print(f"Segmentation method: Advanced arc segmentation")
    print("Parameters used:")
    for key, value in segmentation_params.items():
        print(f"  {key}: {value}")
    
    return model, scaler, cv_mean, segmentation_params

if __name__ == "__main__":
    import sys
    
    print("=== PYTORCH LSTM CLASSIFIER SUITE ===\n")
    print("Available modes:")
    print("  1. PyTorch LSTM only training")
    
    result = main_lstm_training()
    
    if result:
        model, scaler, cv_accuracy, segmentation_params = result
        
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"PyTorch LSTM model trained successfully!")
        print(f"Cross-validation accuracy: {cv_accuracy:.4f}")
        print("Model and scaler saved for future use.")
        print("Segmentation used optimized parameters:")
        for key, value in segmentation_params.items():
            print(f"  {key}: {value}")
        print("Model ready for deployment and inference.")
    else:
        print("Training failed. Please check your configuration and data.")