"""
cnn_math.py

This module provides utility functions for a Convolutional Neural Network (CNN) implementation, including:

- Activation functions (ReLU and Sigmoid) and their derivatives
- Loss functions (Binary Cross-Entropy) and their derivatives
- Functions to calculate loss and activation based on provided type

These functions are used in the forward and backward propagation steps of the CNN.
"""

import numpy as np

# Activation functions and their derivatives ----------------------------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Loss functions and their derivatives ----------------------------------------------------------------------------------------

def binary_crossentropy_loss(y_true, y_pred):
    epsilon = 1e-7  # to prevent division by zero
    return -np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)) / y_true.shape[0]

def binary_crossentropy_loss_derivative(y_true, y_pred):
    epsilon = 1e-7  # to prevent division by zero
    return (y_pred - y_true) / ((y_pred + epsilon) * (1 - y_pred + epsilon) * y_true.shape[0])

# Functions to calculate loss and activation based on provided type ----------------------------------------------------------

def calculate_loss(y_pred, y_true, loss_type):
    if loss_type == 'binary_crossentropy':
        return binary_crossentropy_loss(y_true, y_pred)
    
    raise ValueError(f'Loss type {loss_type} not supported')

def calculate_loss_gradient(y_pred, y_true, loss_type):
    if loss_type == 'binary_crossentropy':
        return binary_crossentropy_loss_derivative(y_true, y_pred)
    
    raise ValueError(f'Loss type {loss_type} not supported')

def calculate_activation_func(x, activation_func_type):
    if activation_func_type == 'relu':
        return relu(x)
    
    if activation_func_type == 'sigmoid':
        return sigmoid(x)
    
    raise ValueError(f'Activation function {activation_func_type} not supported')

def calculate_activation_func_derivative(x, activation_func_type):
    if activation_func_type == 'relu':
        return relu_derivative(x)
    
    if activation_func_type == 'sigmoid':
        return sigmoid_derivative(x)
    
    raise ValueError(f'Activation function {activation_func_type} not supported')