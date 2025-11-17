#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: hipot_sin2.py
Author: Javier del Río
Date: 2025-09-26
Description: 
    Sine-squared (sin²) model fitting utilities for RFID signal analysis.
    Implements parametric and normalized fitting of y = a * sin²(b*x + c) + d
    models to experimental data with curve optimization and visualization.
    Useful for modeling periodic RFID signal patterns and antenna responses.

License: MIT License
Dependencies: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sin2_model(x, a, b, c, d):
    """
    Sine-squared model function.
    
    :param x: Independent variable (time or position)
    :param a: Amplitude parameter
    :param b: Angular frequency parameter
    :param c: Phase shift parameter
    :param d: Vertical offset parameter
    :return: y = a * sin²(b*x + c) + d
    """
    return a * np.sin(b * x + c)**2 + d

def sin2_model_normalized(x, b, c):
    """
    Normalized sine-squared model function without amplitude and offset parameters.
    
    :param x: Independent variable (time or position)
    :param b: Angular frequency parameter
    :param c: Phase shift parameter
    :return: y = sin²(b*x + c)
    """
    return np.sin(b * x + c)**2

def fit_sin2_model(x_data, y_data):
    """
    Fits the complete sine-squared model to the data.
    
    :param x_data: Array of x values
    :param y_data: Array of y values
    :return: Fitted parameters [a, b, c, d]
    """
    # Initial parameter estimates
    initial_guess = [
        (np.max(y_data) - np.min(y_data)) / 2,  # a: approximate amplitude
        2 * np.pi / (x_data[-1] - x_data[0]),   # b: approximate angular frequency
        np.arcsin(x_data[0]),                   # c: phase shift
        np.mean(y_data)                         # d: vertical offset
    ]
    
    # Fit model to data
    params, covariance = curve_fit(
        sin2_model, x_data, y_data, p0=initial_guess, maxfev=10000
    )
    
    return params       

def fit_sin2_model_normalized(x_data, y_data):
    """
    Fits the normalized sine-squared model to the data.
    
    :param x_data: Array of x values
    :param y_data: Array of y values (should be normalized 0-1)
    :return: Fitted parameters [b, c]
    """
    # Initial parameter estimates
    initial_guess = [
        2 * np.pi / (x_data[-1] - x_data[0]),  # b: approximate angular frequency
        np.arcsin(x_data[1])                   # c: phase shift
    ]

    # Fit model to data
    params, covariance = curve_fit(sin2_model_normalized, x_data, y_data, maxfev=100000)
    
    return params

def plot_fit(x_data, y_data, params):
    """
    Plots the data and fitted model.
    
    :param x_data: Array of x values
    :param y_data: Array of y values
    :param params: Fitted parameters [a, b, c, d]
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(x_data, y_data, label='Data', color='blue')
    
    # Generate values for fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = sin2_model(x_fit, *params)
    
    plt.plot(x_fit, y_fit, label='Fit: $y = {:.2f} \cdot \sin^2({:.2f} \cdot x + {:.2f}) + {:.2f}$'.format(*params), color='red')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sine-Squared Model Fitting')
    plt.legend()
    plt.grid()
    plt.show()

def plot_fit_normalized(x_data, y_data, params):
    """
    Plots the data and fitted normalized model.
    
    :param x_data: Array of x values
    :param y_data: Array of normalized y values
    :param params: Fitted parameters [b, c]
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(x_data, y_data, label='Data', color='blue')
    
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = sin2_model_normalized(x_fit, *params)
    
    plt.plot(x_fit, y_fit, label='Normalized fit: $y = \sin^2({:.2f} \cdot x + {:.2f})$'.format(*params), color='green')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Normalized Sine-Squared Model Fitting')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Example data generation
    x_data = np.linspace(0, 10, 1000).astype(np.float64)
    y_data = (2 * np.sin(1.5 * x_data + 0.5)**2 + 1 + np.random.normal(0, 0.1, x_data.size)).astype(np.float64)  # Add noise
    
    # Fit the model to the data
    params = fit_sin2_model(x_data, y_data)
    
    # Plot the data and fitted model
    plot_fit(x_data, y_data, params)

    # Example data for normalized model
    y_data = 2 * np.sin(1.5 * x_data + 0.5)**2 + 1 + np.random.normal(0, 0.1, x_data.size)  # Add noise
    y_data_norm = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))  # Normalize data

    # Fit the normalized model to the data
    params_norm = fit_sin2_model_normalized(x_data, y_data_norm)
    
    # Plot the data and fitted normalized model
    plot_fit_normalized(x_data, y_data_norm, params_norm)


