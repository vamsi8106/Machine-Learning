import numpy as np

def sigmoid(x):
    """
    Computing the sigmoid function for input x.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray : Sigmoid of x.
    """
    x = np.clip(x, -88.72, 88.72)  # Clip x to prevent overflow in exponential function for float32 precision
    return 1 / (1 + np.exp(-x))  # Sigmoid function: 1 / (1 + e^(-x))

def accuracy(y_pred, y):
    """
    Computimg accuracy of predictions compared to true labels.

    Parameters:
    y_pred (numpy.ndarray): Predicted labels.
    y (numpy.ndarray): True labels.

    Returns:
    float: Accuracy as a ratio of correct predictions to total predictions.
    """
    return np.sum(y_pred == y) / len(y)  # Compute accuracy as the ratio of correct predictions to total predictions
