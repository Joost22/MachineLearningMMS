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
weerdata2 = data['precip']

# min_weerdata = np.min(weerdata)
# #print(min_weerdata)
# max_weerdata = np.max(weerdata)
# #print(max_weerdata)

# #data.plot(kind='scatter', x='weerdata', y='Taxi drives per hour', figsize=(12,8))

# rounded_weerdatas = np.zeros(data_points)
# for i in range(data_points):
#     rounded_weerdata = np.round(weerdata[i] * 2) / 2
#     rounded_weerdatas[i] = rounded_weerdata

# data['roundedweerdata'] = rounded_weerdatas

# round_min_weerdata = round(min_weerdata * 2) / 2
# round_max_weerdata = round(max_weerdata * 2) / 2
# temp_points = np.arange(round_min_weerdata, round_max_weerdata, 0.5)
# #print(temp_points)

# average_drives_temp = np.zeros([len(temp_points),2])
# for i in range(len(temp_points)):
#     drives_temp = []
#     for j in range(data_points):
#         if data['roundedweerdata'][j] == temp_points[i]:
#             drives_temp.append(data['Taxi drives per hour'][j])
#     average_drives = np.average(drives_temp)
#     average_drives_temp[i] = [temp_points[i], average_drives]
    
# #print(average_drives_temp)

# rounded_weerdatas_2 = np.zeros(data_points)
# for i in range(data_points):
#     rounded_weerdata = np.round(weerdata[i])
#     rounded_weerdatas_2[i] = rounded_weerdata

# data['roundedweerdata_2'] = rounded_weerdatas_2

# round_min_weerdata_2 = round(min_weerdata)
# round_max_weerdata_2 = round(max_weerdata)
# temp_points_2 = np.arange(round_min_weerdata_2, round_max_weerdata_2, 1)

# average_drives_temp_2 = np.zeros([len(temp_points_2),2])
# for i in range(len(temp_points_2)):
#     drives_temp_2 = []
#     for j in range(data_points):
#         if data['roundedweerdata_2'][j] == temp_points_2[i]:
#             drives_temp_2.append(data['Taxi drives per hour'][j])
#     average_drives_2 = np.average(drives_temp_2)
#     average_drives_temp_2[i] = [temp_points_2[i], average_drives_2]
    
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

data_weerdata = data.iloc[:,[3,7,13,16,-1]]
#print(data_weerdata)

data_new = data[['windspeed']]
#print(data_new)
#data_new[np.isnan(data_new)] = 0
#data_new.drop(data_new[data_new[1] <= 0.001].index, inplace = True)
#print(data_new)

#data_new.insert(0, 'Ones', 1)

#Initializing the variables, setting X (training data) and y (target variable)
#cols = data_new.shape[1]
#print(cols)
#X = data_new.iloc[:,[1,2,3,4]]
#print('x is ',X)
#y = data_new.iloc[:,[-1]]
#print(y)

X = data_new
y = data[['Taxi drives per hour']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # The cost function is expecting numpy matrices. Therefore, convert X and y.
X_matrix = X_train
print(X_matrix)
# y_matrix = np.matrix(y.values)
# #print(y)

# Implement the linear regression with scikit-learn below!
model = linear_model.LinearRegression()
reg = model.fit(X_train,y_train)
# coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
# coeff_df
print(model.coef_)

y_pred = model.predict(X_test)
print(model.score(X_test,y_test))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Plot the results 
#x = np.array(X[:, 1].A1)
x = np.array(X_matrix[:,0].A1)
#print(x)
f = y_pred
#print(f)
fig, ax = plt.subplots(figsize=(12,8))
ax.set_ylim([0, 15000])
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(X['temp'], y, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Weerdata')
ax.set_ylabel('Taxi Drives')
ax.set_title('Predicted Taxi Drives vs. Weerdata')