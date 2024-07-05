import numpy as np

class LinearRegression():

    def __init__(self,lr=0.001,n_iterations=1000):

        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self,X,y):

        n_samples,n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):

            y_pred = np.dot(X , self.weights) + self.bias

            dw = 2 * (1/n_samples) * np.dot(X.T,(y_pred - y))
            db = 2 * (1/n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db


    def predict(self,X):

        return np.dot(X,self.weights) + self.bias
