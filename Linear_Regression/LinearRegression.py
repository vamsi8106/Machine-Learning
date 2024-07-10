import numpy as np

class LinearRegression():

    def __init__(self, lr=0.001, n_iterations=1000):
        """
        Initializing Linear Regression model.

        Parameters:
        lr (float): Learning rate for gradient descent. Default is 0.001.
        n_iterations (int): Number of iterations for gradient descent. Default is 1000.
        """
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None  # Initializing weights to None
        self.bias = None  # Initializing bias to None

    def fit(self, X, y):
        """
        Fit the Linear Regression model to the training data.

        Parameters:
        X (numpy.ndarray): Training samples, shape (n_samples, n_features).
        y (numpy.ndarray): Target values, shape (n_samples,).

        Returns:
        None
        """
        n_samples, n_features = X.shape  # Number of samples and features

        self.weights = np.zeros(n_features)  # Initializing weights to zeros
        self.bias = 0  # Initializing bias to 0

        for _ in range(self.n_iterations):
            # Predicted values with current weights and bias
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradient of mean squared error loss function with repect to weights and bias
            dw = 2 * (1 / n_samples) * np.dot(X.T, (y_pred - y))  # Gradient of weights
            db = 2 * (1 / n_samples) * np.sum(y_pred - y)  # Gradient of bias

            # Update weights and bias using gradient descent
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        """
        Predict target values for input samples.

        Parameters:
        X (numpy.ndarray): Input samples, shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Predicted target values, shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias
