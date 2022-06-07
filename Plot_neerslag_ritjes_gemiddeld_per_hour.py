#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:32:39 2022

@author: PieterClemens
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import os
import math

#print(os.getcwd())
path = os.getcwd() + '\Weather_TaxiDrivesPerHour.csv'
data = pd.read_csv(path, header=0)
#print(data)
#print(data.head())

data.describe()
data_points = len(data.iloc[:,1])
#print(data_points)

weerdata = data['temp']

min_weerdata = np.min(weerdata)
#print(min_weerdata)
max_weerdata = np.max(weerdata)
#print(max_weerdata)

#data.plot(kind='scatter', x='weerdata', y='Taxi drives per hour', figsize=(12,8))

rounded_weerdatas = np.zeros(data_points)
for i in range(data_points):
    rounded_weerdata = np.round(weerdata[i] * 2) / 2
    rounded_weerdatas[i] = rounded_weerdata

data['roundedweerdata'] = rounded_weerdatas

round_min_weerdata = round(min_weerdata * 2) / 2
round_max_weerdata = round(max_weerdata * 2) / 2
temp_points = np.arange(round_min_weerdata, round_max_weerdata, 0.5)
#print(temp_points)

average_drives_temp = np.zeros([len(temp_points),2])
for i in range(len(temp_points)):
    drives_temp = []
    for j in range(data_points):
        if data['roundedweerdata'][j] == temp_points[i]:
            drives_temp.append(data['Taxi drives per hour'][j])
    average_drives = np.average(drives_temp)
    average_drives_temp[i] = [temp_points[i], average_drives]
    
#print(average_drives_temp)

rounded_weerdatas_2 = np.zeros(data_points)
for i in range(data_points):
    rounded_weerdata = np.round(weerdata[i])
    rounded_weerdatas_2[i] = rounded_weerdata

data['roundedweerdata_2'] = rounded_weerdatas_2

round_min_weerdata_2 = round(min_weerdata)
round_max_weerdata_2 = round(max_weerdata)
temp_points_2 = np.arange(round_min_weerdata_2, round_max_weerdata_2, 1)

average_drives_temp_2 = np.zeros([len(temp_points_2),2])
for i in range(len(temp_points_2)):
    drives_temp_2 = []
    for j in range(data_points):
        if data['roundedweerdata_2'][j] == temp_points_2[i]:
            drives_temp_2.append(data['Taxi drives per hour'][j])
    average_drives_2 = np.average(drives_temp_2)
    average_drives_temp_2[i] = [temp_points_2[i], average_drives_2]
    
#print(average_drives_temp_2)

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(average_drives_temp[:,0], average_drives_temp[:,1], 'r', label='rounded to heel or half')
# ax.plot(average_drives_temp_2[:,0], average_drives_temp_2[:,1], 'b', label='rounded to heel')
# ax.legend(loc=4)
# ax.set_xlabel('Weerdata')
# ax.set_ylabel('Taxi Drives')
# ax.set_title('Taxi Drives vs. weerdata temperature')

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(average_drives_temp[:,0], average_drives_temp[:,1])
# ax.scatter(average_drives_temp_2[:,0], average_drives_temp_2[:,1])

#print(data['weerdata'])

data_new = pd.DataFrame(average_drives_temp)
data_new[np.isnan(data_new)] = 0
data_new.drop(data_new[data_new[1] <= 0.001].index, inplace = True)
print(data_new)

data_new.insert(0, 'Ones', 1)

#Initializing the variables, setting X (training data) and y (target variable)
#cols = data_new.shape[1]
#print(cols)
X = data_new.iloc[:,[0,1]]
print(X)
y = data_new.iloc[:,[2]]
print(y)

# The cost function is expecting numpy matrices. Therefore, convert X and y.
X_matrix = np.matrix(X.values)
#print(X)
y_matrix = np.matrix(y.values)
#print(y)

# Implement the linear regression with scikit-learn below!
reg = linear_model.LinearRegression().fit(X_matrix,y_matrix)

# Plot the results 
#x = np.array(X[:, 1].A1)
x = np.array(X_matrix[:, 1].A1)
#print(x)
f = reg.predict(X_matrix).flatten()
#print(f)
fig, ax = plt.subplots(figsize=(12,8))
ax.set_ylim([0, 15000])
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(X[:][0], y, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Weerdata')
ax.set_ylabel('Taxi Drives')
ax.set_title('Predicted Taxi Drives vs. Weerdata')