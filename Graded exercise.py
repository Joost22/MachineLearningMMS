# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:08:34 2022

@author: roosm
"""
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv('Weather_TaxiDrives.csv')

data.plot(kind='scatter', x='temp', y='Taxi Drives', figsize=(12,8))

# Add a column of ones to the training set so we can use a vectorized solution to computing the cost and gradients.
data.insert(0, 'Ones', 1)

#Initializing the variables, setting X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,[0,6]]
y = data.iloc[:,cols-1:cols]

# The cost function is expecting numpy matrices. Therefore, convert X and y.
X = np.matrix(X.values)
y = np.matrix(y.values)

# Implement the linear regression with scikit-learn below!
reg = linear_model.LinearRegression().fit(X,y)

# Plot the results 
x = np.array(X[:, 1].A1)
f = reg.predict(X).flatten()
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.iloc[:,[6]], data.iloc[:,[35]], label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Taxi Drives')
ax.set_ylabel('Temperature')
ax.set_title('Predicted Temperature vs. Taxi Drives')