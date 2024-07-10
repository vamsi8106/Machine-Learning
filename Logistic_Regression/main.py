# Import necessary libraries and modules
from sklearn.datasets import load_breast_cancer
from LogisticRegression import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from utils import *

# Loading breast cancer dataset (binary classification)
data = load_breast_cancer()

# Separating features and target
X = data.data  # Features
y = data.target  # Target (0 or 1)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing logistic regression model
logistic_model = LogisticRegression()

# Training the logistic regression model on the training data
logistic_model.fit(X_train, y_train)

# Predicting the labels for the test set
y_pred = logistic_model.predict(X_test)

# Calculating and printing the accuracy of the model
print(accuracy(y_pred, y_test))
