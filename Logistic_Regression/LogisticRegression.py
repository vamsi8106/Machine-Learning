import numpy as np
from utils import sigmoid

class LogisticRegression:
    def __init__(self, lr=0.01, n_iterations=1000):
        """
        Initializing logistic regression model.

        Parameters:
        lr (float): Learning rate for gradient descent.
        n_iterations (int): Number of iterations for gradient descent.
        """
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fitting the logistic regression model to the training data.

        Parameters:
        X (numpy.ndarray): Training features, shape (n_samples, n_features).
        y (numpy.ndarray): Target labels, shape (n_samples,).

        """
        n_samples, n_features = X.shape
        
        # Initializing weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Linear combination of weights and input features
            y_pred = np.dot(X, self.weights) + self.bias
            # Applying sigmoid activation function
            y_pred = sigmoid(y_pred)

            # Computing gradients of binary cross entropy loss function with respect to weights and bias.
            dw = 2 * (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = 2 * (1 / n_samples) * np.sum(y_pred - y)

            # Updating weights and bias using gradients and learning rate
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predicting class labels for input data.

        Parameters:
        X (numpy.ndarray): Input features for prediction, shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Predicted class labels (binary), shape (n_samples,).
        """
        # Linear combination of weights and input features
        y_pred = np.dot(X, self.weights) + self.bias
        # Applying sigmoid activation function
        y_pred = sigmoid(y_pred)
        # Convert predicted probabilities to binary class labels
        y_pred_class = np.where(y_pred > 0.5, 1, 0)

        return y_pred_class
