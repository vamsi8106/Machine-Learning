import numpy as np

def sigmoid(x):
    x = np.clip(x, -88.72, 88.72)
    return (1/(1 + np.exp(-x)))

def accuracy(y_pred, y):

    return np.sum(y_pred == y) / len(y)
