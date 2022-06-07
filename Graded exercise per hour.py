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
import math

data = pd.read_csv('Weather_TaxiDrivesPerHour.csv')

data_points = len(data.iloc[:,1])
min_temp = np.min(data['temp'])
max_temp = np.max(data['temp'])

data.plot(kind='scatter', x='temp', y='Taxi drives per hour', figsize=(12,8))

rounded_temp = np.zeros(data_points)
for i in range(data_points):
    rounded_temp1 = np.round(data['temp'][i] * 2) / 2
    rounded_temp[i] = rounded_temp1

data['roundedtemp'] = rounded_temp

round_min_temp = round(min_temp * 2) / 2
round_max_temp = round(max_temp * 2) / 2
temp_points = np.arange(round_min_temp, round_max_temp, 0.5)

average_drives_temp = np.zeros([len(temp_points),2])
for i in range(len(temp_points)):
    drives_temp = []
    for j in range(data_points):
        if data['roundedtemp'][j] == temp_points[i]:
            drives_temp.append(data['Taxi drives per hour'][j])
    average_drives = np.average(drives_temp)
    average_drives_temp[i] = [temp_points[i], average_drives]

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(average_drives_temp[:,0], average_drives_temp[:,1], 'r', label='Taxi Drives')

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(average_drives_temp[:,0], average_drives_temp[:,1])

data_new = pd.DataFrame(average_drives_temp)
data_new[np.isnan(data_new)] = 0

# Add a column of ones to the training set so we can use a vectorized solution to computing the cost and gradients.
data_new.insert(0, 'Ones', 1)

#Initializing the variables, setting X (training data) and y (target variable)
cols = data_new.shape[1]
X = data_new.iloc[:,[0, 1]]
y = data_new.iloc[:,[2]]

##Initializing the variables, setting X (training data) and y (target variable)
#cols = data.shape[1]
#X = data_new.iloc[:,[0,36]]
#y = data_new.iloc[:,[35]]

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
ax.scatter(data_new.iloc[:,[1]], data_new.iloc[:,[2]], label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Taxi Drives')
ax.set_ylabel('Temperature')
ax.set_title('Predicted Temperature vs. Taxi Drives')