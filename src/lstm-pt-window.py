#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: lstm-pt-window.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    LSTM neural network implementation using PyTorch with sliding window approach
    for RFID tag presence detection. Processes time-series data from multiple CSV files
    to classify dynamic vs static scenarios using normalized features and GPU acceleration.

License: MIT License
Dependencies: numpy, torch, sklearn, joblib, feature_extraction (local), read_csv (local)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from csv_data_loader import extract_tag_data
from feature_extraction import normalize_features

# Load data from CSV files
csv_file_dynamic = 'data/dynamic.csv'
csv_file_static = 'data/static.csv'
csv_file_magic = 'data/magic_mike.csv'

tags_data_dynamic = extract_tag_data(csv_file_dynamic)
tags_data_static = extract_tag_data(csv_file_static)
tags_data_magic = extract_tag_data(csv_file_magic)

# Select one tag from each file (you can adapt this according to your needs)
tag_data_dynamic = tags_data_dynamic[list(tags_data_dynamic.keys())[0]]
tag_data_static = tags_data_static[list(tags_data_static.keys())[0]]
tag_data_magic1 = tags_data_magic[list(tags_data_magic.keys())[0]]

# Extract complete arrays from each tag
def get_arrays(tag_data):
    return tag_data['rssi'], tag_data['phase'], tag_data['timestamp']

rssi_dyn, phase_dyn, ts_dyn = get_arrays(tag_data_dynamic)
rssi_stat, phase_stat, ts_stat = get_arrays(tag_data_static)
rssi_magic, phase_magic, ts_magic = get_arrays(tag_data_magic1)

# Labels: 1 for dynamic and magic (present), 0 for static (absent)
labels_dyn = np.ones(len(rssi_dyn), dtype=np.float32)
labels_stat = np.zeros(len(rssi_stat), dtype=np.float32)
labels_magic = np.ones(len(rssi_magic), dtype=np.float32)

# Concatenate all data
rssi = np.concatenate([rssi_dyn, rssi_stat, rssi_magic])
phase = np.concatenate([phase_dyn, phase_stat, phase_magic])
timestamp = np.concatenate([ts_dyn, ts_stat, ts_magic])
labels = np.concatenate([labels_dyn, labels_stat, labels_magic])

# Build feature matrix per sample
features = np.stack([
    rssi,
    phase,
    np.diff(np.insert(timestamp, 0, timestamp[0]))  # Time differences
], axis=1)

# Normalize features
features_list = [dict(zip(['rssi','phase','dt'], row)) for row in features]
features_normalized, scaler = normalize_features(features_list)
features_normalized = np.array(features_normalized, dtype=np.float32)

# Create sequences with sliding window
seq_len = 300
X = []
y = []
for i in range(len(features_normalized) - seq_len):
    X.append(features_normalized[i:i+seq_len])
    y.append(labels[i+seq_len-1])  # Label corresponds to the last element of the window
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- GPU USAGE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
X_train_tensor = torch.tensor(X_train).to(device)
y_train_tensor = torch.tensor(y_train).unsqueeze(-1).to(device)
X_test_tensor = torch.tensor(X_test).to(device)
y_test_tensor = torch.tensor(y_test).unsqueeze(-1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# LSTM model in PyTorch
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
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

model = LSTMClassifier(input_dim=X.shape[2]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 50
print("Starting LSTM model training...")
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
print("Evaluating model performance...")
model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        all_preds.extend((preds.cpu().numpy() > 0.5).astype(int).flatten())
        all_true.extend(yb.cpu().numpy().flatten())

print("\n=== MODEL PERFORMANCE REPORT ===")
print(classification_report(all_true, all_preds))

# Save model and scaler
print("Saving trained model and feature scaler...")
torch.save(model.state_dict(), 'lstm_model_pytorch_full.pt')
import joblib
joblib.dump(scaler, 'scaler_full.pkl')
print("PyTorch LSTM model trained and saved successfully.")

# Print model summary
print(f"\n=== MODEL CONFIGURATION ===")
print(f"Input features: {X.shape[2]} (RSSI, Phase, Time difference)")
print(f"Sequence length: {seq_len} samples")
print(f"Hidden dimensions: 100")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Device used: {device}")
print(f"Model saved to: lstm_model_pytorch_full.pt")
print(f"Scaler saved to: scaler_full.pkl")