from sklearn.datasets import load_breast_cancer
from LogisticRegression import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from utils import *

# Load breast cancer dataset (binary classification)
data = load_breast_cancer()

# Separate features and target
X = data.data  # Features
y = data.target  # Target (0 or 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression()

logistic_model.fit(X_train,y_train)

y_pred = logistic_model.predict(X_test)

print(accuracy(y_pred,y_test))
