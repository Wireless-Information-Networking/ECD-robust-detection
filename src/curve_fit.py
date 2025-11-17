#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: curve_fit.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Polynomial curve fitting utilities for RFID tag RSSI data analysis.
    Provides functions to fit polynomial models of arbitrary degree to RSSI
    measurements with parameter uncertainty analysis and model comparison.
    Includes visualization tools for comparing different polynomial degrees
    and statistical metrics for model selection and validation.

License: MIT License
Dependencies: numpy, scipy, matplotlib, typing, csv_data_loader (local)
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
import os



def polynomial_model(x: np.ndarray, *params: float) -> np.ndarray:
    """
    Polynomial model of arbitrary degree.
    
    :param x: Independent variable array
    :param params: Polynomial coefficients (a0, a1, a2, ...)
    :return: Polynomial evaluation y = a0 + a1*x + a2*x^2 + ...
    """
    return sum(p * x**i for i, p in enumerate(params))

def fit_polynomial(x_data: np.ndarray, y_data: np.ndarray, degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits a polynomial model to the data and returns the parameters and covariance.
    
    :param x_data: Independent variable data
    :param y_data: Dependent variable data  
    :param degree: Polynomial degree
    :return: Tuple of (fitted parameters, covariance matrix)
    """
    # Initial parameter estimates (all zeros)
    initial_guess = np.zeros(degree + 1)

    # Fit the polynomial model to the data
    params, covariance = curve_fit(polynomial_model, x_data, y_data, p0=initial_guess)
    
    return params, covariance

def plot_polynomial_fit(x_data: np.ndarray, y_data: np.ndarray, params: np.ndarray) -> None:
    """
    Plots the original data and the fitted polynomial model.
    
    :param x_data: Independent variable data
    :param y_data: Dependent variable data
    :param params: Fitted polynomial parameters
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(x_data, y_data, label='Original Data', color='blue', alpha=0.6)

    # Generate smooth curve for fitted polynomial
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = polynomial_model(x_fit, *params)

    plt.plot(x_fit, y_fit, label='Polynomial Fit', color='red', linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('RSSI (dBm)')
    plt.title('Polynomial Model Fitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def fit_and_plot_polynomial(x_data: List[float], y_data: List[float], degree: int) -> None:
    """
    Fits a polynomial model to the data and plots the result with parameters.
    
    :param x_data: Independent variable data as list
    :param y_data: Dependent variable data as list
    :param degree: Polynomial degree to fit
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Fit the polynomial model
    params, covariance = fit_polynomial(x_data, y_data, degree)

    # Plot the fitted model
    plot_polynomial_fit(x_data, y_data, params)

    print(f"Parameters of degree {degree} polynomial model: {params}")
    print(f"Parameter covariance matrix: {covariance}")

if __name__ == "__main__":
    # Load and process RFID data
    # Extract data from the dynamic.csv file and analyze first tag up to t = 1.3s
    
    # Add src directory to Python path for module imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    # Now import using absolute import
    from csv_data_loader import extract_tag_data, normalize_timestamps

    data = extract_tag_data('data/dynamic.csv')
    normalized_data = normalize_timestamps(data)
    tag_id = list(data.keys())[0]
    tag_data = data[tag_id]

    # Select data range for analysis
    init = 280
    ending = 695
    x_data = tag_data['timestamp'][init:ending]
    y_data = tag_data['rssi'][init:ending]
    
    # Display polynomial fits of different degrees
    plt.figure(figsize=(12, 6))
    plt.scatter(x_data, y_data, label='Original Data', color='blue', alpha=0.6, s=15)

    # Fit and plot polynomials of degrees 2 through 7
    plt.title(f'Polynomial Fits Comparison for Tag {tag_id[:8]}')

    # Generate smooth curve points for visualization
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
    for i, degree in enumerate(range(2, 8)):
        params, _ = fit_polynomial(x_data, y_data, degree)
        y_fit = polynomial_model(x_fit, *params)
        plt.plot(x_fit, y_fit, label=f'Degree {degree}', color=colors[i], linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('RSSI (dBm)')
    plt.title('Polynomial Fits of Different Degrees')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Statistical analysis: parameter uncertainty and model performance
    degrees = list(range(2, 8))
    param_errors = []
    mses = []
    
    for degree in degrees:
        params, covariance = fit_polynomial(x_data, y_data, degree)
        # Calculate standard errors from covariance matrix diagonal
        std_errors = np.sqrt(np.diag(covariance))
        param_errors.append(np.mean(std_errors))
        
        # Calculate Mean Squared Error for model performance
        y_pred = polynomial_model(x_data, *params)
        mse = np.mean((y_data - y_pred) ** 2)
        mses.append(mse)

    # Visualization: parameter uncertainty vs model performance
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot mean parameter uncertainty
    color = 'tab:blue'
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('Mean Parameter Uncertainty', color=color)
    ax1.plot(degrees, param_errors, marker='o', color=color, label='Mean Uncertainty', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Mean Squared Error on secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Squared Error (MSE)', color=color)
    ax2.plot(degrees, mses, marker='s', color=color, label='MSE', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Model Performance Analysis: Parameter Uncertainty vs MSE')
    fig.tight_layout()  # Prevent axis overlap
    plt.show()

    # Print numerical results
    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    print(f"{'Degree':<8} {'Mean Param Error':<18} {'MSE':<12}")
    print("-" * 40)
    for i, degree in enumerate(degrees):
        print(f"{degree:<8} {param_errors[i]:<18.6f} {mses[i]:<12.6f}")

