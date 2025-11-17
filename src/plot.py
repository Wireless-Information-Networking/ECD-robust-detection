#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: plot.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Visualization script for RFID tag data from CSV files. Provides functions
    to plot raw and normalized RSSI/Phase data with customizable markers and
    optional polynomial fitting for analysis and presentation purposes.

License: MIT License
Dependencies: numpy, matplotlib, read_csv (local), curve_fit (local)
"""

import numpy as np
import matplotlib.pyplot as plt

from csv_data_loader import extract_tag_data, normalize_timestamps

def plot_tag_data(tag_data):
    """
    Plots tag data with small points for better visibility.
    
    :param tag_data: Dictionary containing tag data with 'timestamp', 'rssi', 'phase'
    """
    for tag_id, values in tag_data.items():
        plt.figure(figsize=(10, 5))
        plt.plot(values['timestamp'], values['rssi'], '.', label='RSSI', color='blue', markersize=2)
        plt.plot(values['timestamp'], values['phase'], '.', label='Phase', color='orange', markersize=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.title(f'Tag ID: {tag_id}')
        plt.legend()
        plt.grid()
        plt.show()

def plot_normalized_data(tag_data):
    """
    Plots normalized tag data with small points for better visibility.
    
    :param tag_data: Dictionary containing tag data with 'timestamp', 'rssi', 'phase'
    """
    for tag_id, values in tag_data.items():
        # Normalize the data
        rssi_norm = (values['rssi'] - np.min(values['rssi'])) / (np.max(values['rssi']) - np.min(values['rssi']))
        phase_norm = (values['phase'] - np.min(values['phase'])) / (np.max(values['phase']) - np.min(values['phase']))
        
        plt.figure(figsize=(10, 5))
        plt.plot(values['timestamp'], rssi_norm, '.', label='Normalized RSSI', color='blue', markersize=2)
        plt.plot(values['timestamp'], phase_norm, '.', label='Normalized Phase', color='orange', markersize=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Values')
        plt.title(f'Normalized Data for Tag ID: {tag_id}')
        plt.legend()
        plt.grid()
        plt.show()

def plot_rssi_phase_separate(tag_data, save_plots=False, output_dir='output_plots'):
    """
    Plots RSSI and Phase in separate subplots for better analysis.
    
    :param tag_data: Dictionary containing tag data
    :param save_plots: Whether to save plots to files
    :param output_dir: Directory to save plots if save_plots is True
    """
    import os
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    for tag_id, values in tag_data.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # RSSI subplot
        ax1.plot(values['timestamp'], values['rssi'], '.', color='blue', markersize=1.5)
        ax1.set_ylabel('RSSI (dBm)')
        ax1.set_title(f'Tag ID: {tag_id} - RSSI Signal')
        ax1.grid(True, alpha=0.3)
        
        # Phase subplot
        ax2.plot(values['timestamp'], values['phase'], '.', color='orange', markersize=1.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Phase')
        ax2.set_title(f'Tag ID: {tag_id} - Phase Signal')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'tag_{tag_id}_signals.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        
        plt.show()

def plot_tag_statistics(tag_data):
    """
    Plots statistical information about the tag data.
    
    :param tag_data: Dictionary containing tag data
    """
    for tag_id, values in tag_data.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RSSI histogram
        axes[0,0].hist(values['rssi'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].set_xlabel('RSSI (dBm)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title(f'RSSI Distribution\nMean: {np.mean(values["rssi"]):.2f} dBm')
        axes[0,0].grid(True, alpha=0.3)
        
        # Phase histogram
        axes[0,1].hist(values['phase'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0,1].set_xlabel('Phase')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title(f'Phase Distribution\nMean: {np.mean(values["phase"]):.2f}')
        axes[0,1].grid(True, alpha=0.3)
        
        # Time differences histogram
        time_diffs = np.diff(values['timestamp'])
        axes[1,0].hist(time_diffs, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1,0].set_xlabel('Time Difference (s)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title(f'Sampling Intervals\nMean: {np.mean(time_diffs):.4f} s')
        axes[1,0].grid(True, alpha=0.3)
        
        # Signal quality over time (RSSI variance in sliding window)
        window_size = max(100, len(values['rssi'])//50)
        rssi_variance = []
        time_centers = []
        
        for i in range(0, len(values['rssi']) - window_size, window_size//2):
            window_rssi = values['rssi'][i:i+window_size]
            rssi_variance.append(np.var(window_rssi))
            time_centers.append(values['timestamp'][i + window_size//2])
        
        axes[1,1].plot(time_centers, rssi_variance, 'r-', linewidth=2)
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('RSSI Variance')
        axes[1,1].set_title('Signal Variability Over Time')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Statistical Analysis - Tag ID: {tag_id}', fontsize=14)
        plt.tight_layout()
        plt.show()

def plot_multiple_tags_comparison(tag_data):
    """
    Plots multiple tags in the same figure for comparison.
    
    :param tag_data: Dictionary containing multiple tag data
    """
    if len(tag_data) < 2:
        print("Need at least 2 tags for comparison plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot RSSI for all tags
    for i, (tag_id, values) in enumerate(tag_data.items()):
        color = colors[i % len(colors)]
        ax1.plot(values['timestamp'], values['rssi'], '.', 
                color=color, markersize=1, alpha=0.7, label=f'Tag {tag_id}')
    
    ax1.set_ylabel('RSSI (dBm)')
    ax1.set_title('RSSI Comparison - All Tags')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot Phase for all tags
    for i, (tag_id, values) in enumerate(tag_data.items()):
        color = colors[i % len(colors)]
        ax2.plot(values['timestamp'], values['phase'], '.', 
                color=color, markersize=1, alpha=0.7, label=f'Tag {tag_id}')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Phase')
    ax2.set_title('Phase Comparison - All Tags')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_signal_quality_analysis(tag_data):
    """
    Performs and plots signal quality analysis including SNR estimation.
    
    :param tag_data: Dictionary containing tag data
    """
    for tag_id, values in tag_data.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Raw signals
        axes[0,0].plot(values['timestamp'], values['rssi'], 'b-', linewidth=0.5)
        axes[0,0].set_ylabel('RSSI (dBm)')
        axes[0,0].set_title('Raw RSSI Signal')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[1,0].plot(values['timestamp'], values['phase'], 'r-', linewidth=0.5)
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Phase')
        axes[1,0].set_title('Raw Phase Signal')
        axes[1,0].grid(True, alpha=0.3)
        
        # Moving averages
        window_size = max(50, len(values['rssi'])//100)
        
        # RSSI moving average
        rssi_smooth = np.convolve(values['rssi'], np.ones(window_size)/window_size, mode='valid')
        time_smooth = values['timestamp'][window_size//2:len(rssi_smooth)+window_size//2]
        
        axes[0,1].plot(values['timestamp'], values['rssi'], 'b-', alpha=0.3, linewidth=0.5, label='Raw')
        axes[0,1].plot(time_smooth, rssi_smooth, 'r-', linewidth=2, label=f'Moving avg (n={window_size})')
        axes[0,1].set_ylabel('RSSI (dBm)')
        axes[0,1].set_title('RSSI with Moving Average')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Phase moving average
        phase_smooth = np.convolve(values['phase'], np.ones(window_size)/window_size, mode='valid')
        
        axes[1,1].plot(values['timestamp'], values['phase'], 'b-', alpha=0.3, linewidth=0.5, label='Raw')
        axes[1,1].plot(time_smooth, phase_smooth, 'r-', linewidth=2, label=f'Moving avg (n={window_size})')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Phase')
        axes[1,1].set_title('Phase with Moving Average')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Signal-to-noise ratio estimation
        rssi_signal = rssi_smooth
        rssi_noise = values['rssi'][window_size//2:len(rssi_smooth)+window_size//2] - rssi_smooth
        
        # SNR calculation (signal power / noise power)
        signal_power = np.mean(rssi_signal**2)
        noise_power = np.mean(rssi_noise**2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        axes[0,2].plot(time_smooth, rssi_signal, 'g-', linewidth=2, label='Signal component')
        axes[0,2].plot(time_smooth, rssi_noise, 'r-', linewidth=1, alpha=0.7, label='Noise component')
        axes[0,2].set_ylabel('RSSI (dBm)')
        axes[0,2].set_title(f'Signal vs Noise\nSNR ≈ {snr_db:.1f} dB')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Spectral analysis (simple frequency content)
        from scipy import signal
        freqs, psd = signal.welch(values['rssi'], fs=1/np.mean(np.diff(values['timestamp'])), nperseg=min(512, len(values['rssi'])//4))
        
        axes[1,2].semilogy(freqs, psd)
        axes[1,2].set_xlabel('Frequency (Hz)')
        axes[1,2].set_ylabel('Power Spectral Density')
        axes[1,2].set_title('RSSI Frequency Content')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Signal Quality Analysis - Tag ID: {tag_id}', fontsize=14)
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function demonstrating various plotting capabilities.
    """
    import os
    
    print("=== RFID TAG DATA VISUALIZATION DEMONSTRATION ===\n")
    
    # Path to example CSV file
    csv_path = os.path.join(
        os.path.dirname(__file__),
        '../data/test/dynamic.csv'  # Change this path to the CSV file you want to analyze
    )
    
    # Check if file exists, otherwise try alternative paths
    alternative_paths = [
        'data/dynamic.csv',
        '../data/test/dynamic.csv',
        'data/static.csv'
    ]
    
    if not os.path.exists(csv_path):
        print(f"Primary file {csv_path} not found, trying alternatives...")
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                csv_path = alt_path
                print(f"Using alternative file: {csv_path}")
                break
        else:
            print("No suitable CSV files found. Please check file paths.")
            return
    
    print(f"Loading data from: {csv_path}")
    
    # Extract tag data
    tag_data = extract_tag_data(csv_path)
    
    if not tag_data:
        print("Could not extract data from CSV file.")
        return
    
    # Normalize timestamps
    tag_data = normalize_timestamps(tag_data)
    
    print(f"Found {len(tag_data)} tag(s) in the file:")
    for tag_id, values in tag_data.items():
        duration = values['timestamp'][-1] - values['timestamp'][0] if len(values['timestamp']) > 1 else 0
        print(f"  - Tag {tag_id}: {len(values['timestamp'])} samples, duration: {duration:.2f}s")
    
    # Create various visualizations
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    print("1. Basic tag data plots...")
    plot_tag_data(tag_data)
    
    print("2. Normalized data plots...")
    plot_normalized_data(tag_data)
    
    print("3. Separate RSSI and Phase plots...")
    plot_rssi_phase_separate(tag_data, save_plots=True)
    
    print("4. Statistical analysis...")
    plot_tag_statistics(tag_data)
    
    if len(tag_data) > 1:
        print("5. Multi-tag comparison...")
        plot_multiple_tags_comparison(tag_data)
    
    print("6. Signal quality analysis...")
    plot_signal_quality_analysis(tag_data)
    
    # Optional polynomial fitting demonstration
    try:
        from curve_fit import fit_and_plot_polynomial
        print("7. Polynomial fitting demonstration...")
        
        # Use first tag for demonstration
        first_tag_id = list(tag_data.keys())[0]
        first_tag_data = tag_data[first_tag_id]
        
        # Use a subset of data for fitting (avoid memory issues)
        max_samples = 1000
        if len(first_tag_data['timestamp']) > max_samples:
            indices = np.linspace(0, len(first_tag_data['timestamp'])-1, max_samples, dtype=int)
            x_data = np.array(first_tag_data['timestamp'])[indices]
            y_data = np.array(first_tag_data['rssi'])[indices]
        else:
            x_data = np.array(first_tag_data['timestamp'])
            y_data = np.array(first_tag_data['rssi'])
        
        # Fit polynomial (degree 3)
        fit_and_plot_polynomial(x_data, y_data, degree=3)
        
    except ImportError:
        print("Polynomial fitting module not available, skipping...")
    except Exception as e:
        print(f"Error in polynomial fitting: {e}")
    
    print("\n=== VISUALIZATION COMPLETE ===")
    print("All plots have been generated and saved to 'output_plots/' directory")

if __name__ == '__main__':
    main()
