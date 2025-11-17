#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: lstm-pytorch.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    LSTM neural network implementation using PyTorch for RFID tag presence detection.
    Processes segmented RFID data to classify dynamic vs static scenarios using
    extracted statistical features with binary classification approach.

License: MIT License
Dependencies: numpy, pandas, torch, sklearn, joblib, feature_extraction (local), 
              read_csv (local), timediff (local)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from feature_extraction import extract_features, normalize_features
from csv_data_loader import extract_tag_data
from timediff import split_tag_data_by_time

# Load data from CSV files
csv_file_dynamic = 'data/dynamic.csv'
tags_data_dynamic = extract_tag_data(csv_file_dynamic)
csv_file_static = 'data/static.csv'
tags_data_static = extract_tag_data(csv_file_static)
csv_file_magic = 'data/magic_mike.csv'
tags_data_magic = extract_tag_data(csv_file_magic)

# Select tags from each file
tag_data_dynamic = tags_data_dynamic[list(tags_data_dynamic.keys())[0]]
tag_data_static = tags_data_static[list(tags_data_static.keys())[0]]
tag_data_magic1 = tags_data_magic[list(tags_data_magic.keys())[0]]
tag_data_magic2 = tags_data_magic[list(tags_data_magic.keys())[1]]
tag_data_magic3 = tags_data_magic[list(tags_data_magic.keys())[2]]

# Time threshold for segmentation
th = 4.0  # Threshold for time difference
segments_dynamic = split_tag_data_by_time(tag_data_dynamic, threshold=th)
segments_static = split_tag_data_by_time(tag_data_static, threshold=th)
segments_magic1 = split_tag_data_by_time(tag_data_magic1, threshold=th)
segments_magic2 = split_tag_data_by_time(tag_data_magic2, threshold=th)
segments_magic3 = split_tag_data_by_time(tag_data_magic3, threshold=th)

# Combine all segments
segments = segments_dynamic + segments_static + segments_magic1 + segments_magic2 + segments_magic3

# Create labels: 1 for dynamic/magic1 (present), 0 for static/magic2/magic3 (absent)
labels = ([1] * len(segments_dynamic) + 
          [0] * len(segments_static) + 
          [1] * len(segments_magic1) + 
          [0] * len(segments_magic2) + 
          [0] * len(segments_magic3))

# Extract features from all segments
features_list = [extract_features(seg) for seg in segments]
features_normalized, scaler = normalize_features(features_list)

# Prepare data for LSTM
X = np.array(features_normalized, dtype=np.float32)
X = X.reshape((X.shape[0], 1, X.shape[1]))  # (samples, timesteps, features)
y = np.array(labels, dtype=np.float32)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train).unsqueeze(-1)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test).unsqueeze(-1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# LSTM model definition in PyTorch
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=1000):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Use only the last output
        out = self.fc(out)
        return self.sigmoid(out)

# Initialize model and training components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LSTMClassifier(input_dim=X.shape[2]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
print("Starting LSTM model training...")
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    
    # Print progress every 10 epochs
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Model evaluation
print("Evaluating model performance...")
model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        all_preds.extend((preds.cpu().numpy() > 0.5).astype(int).flatten())
        all_true.extend(yb.cpu().numpy().flatten())

print("\n=== MODEL PERFORMANCE REPORT ===")
print(classification_report(all_true, all_preds))

# Save trained model and preprocessing components
print("Saving trained model and preprocessing components...")
torch.save(model.state_dict(), 'lstm_model_pytorch.pt')
import joblib
joblib.dump(scaler, 'scaler.pkl')
np.savez('segments_labels.npz', segments=segments, labels=labels)

print("PyTorch LSTM model trained and saved successfully.")

# Print model summary
print(f"\n=== MODEL CONFIGURATION ===")
print(f"Input features: {X.shape[2]} statistical features per segment")
print(f"Hidden dimensions: 1000")
print(f"Total segments processed: {len(segments)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Device used: {device}")
print(f"Training epochs: {epochs}")

print(f"\n=== SAVED FILES ===")
print(f"Model saved to: lstm_model_pytorch.pt")
print(f"Feature scaler saved to: scaler.pkl")
print(f"Segments and labels saved to: segments_labels.npz")

# Display data distribution
print(f"\n=== DATA DISTRIBUTION ===")
print(f"Dynamic segments (present): {len(segments_dynamic)} + {len(segments_magic1)} = {len(segments_dynamic) + len(segments_magic1)}")
print(f"Static segments (absent): {len(segments_static)} + {len(segments_magic2)} + {len(segments_magic3)} = {len(segments_static) + len(segments_magic2) + len(segments_magic3)}")
print(f"Total segments: {len(segments)}")
print(f"Positive class ratio: {(sum(labels) / len(labels) * 100):.1f}%")