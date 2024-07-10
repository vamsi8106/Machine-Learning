# Importing all necessary libraries
import numpy as np 
from sklearn.model_selection import train_test_split  
from sklearn import datasets 
import matplotlib.pyplot as plt  
import pandas as pd  
from LinearRegression import LinearRegression 
from sklearn.datasets import load_diabetes 

# Load the diabetes dataset from sklearn.datasets
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target  # Assigning features (X) and target (y) from the dataset

# Split the dataset into training and testing sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing a LinearRegression model with learning rate (lr) of 0.1 and 2000 iterations
reg = LinearRegression(lr=0.1, n_iterations=2000)

# Fitting the LinearRegression model on the training data
reg.fit(X_train, y_train)

# Making predictions on the test data using the trained model
predictions = reg.predict(X_test)

# Defining a function to calculate Mean Squared Error (MSE) between y_test and predictions
def mse(y_test, predictions):
    return np.mean((y_test - predictions) ** 2)

# Calculate Mean Squared Error (MSE) between y_test and predictions
mse_value = mse(y_test, predictions)

# Print the calculated MSE value
print("Mean Squared Error (MSE):", mse_value)

# Plotting actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')  # Diagonal line for reference
plt.xlabel('Actual Values')  
plt.ylabel('Predicted Values') 
plt.title('Actual vs Predicted Values')  
plt.grid(True) 
plt.show() 
