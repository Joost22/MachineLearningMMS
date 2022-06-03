#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:32:39 2022

@author: PieterClemens
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.getcwd())
path = os.getcwd() + '\Weather_TaxiDrives_Mathan_3-6.csv'
data = pd.read_csv(path, header=0)
print(data)
print(data.head())

data.describe()
data_points = len(data.iloc[:,1])
print(data_points)

min_feelslike = np.min(data['feelslike'])
print(min_feelslike)
max_feelslike = np.max(data['feelslike'])
print(max_feelslike)

data.plot(kind='scatter', x='feelslike', y='Taxi Drives', figsize=(12,8))

rounded_feelslikes = np.zeros(data_points)
for i in range(data_points):
    rounded_feelslike = np.round(data['feelslike'][i] * 2) / 2
    rounded_feelslikes[i] = rounded_feelslike

data['roundedfeelslike'] = rounded_feelslikes

round_min_feelslike = round(min_feelslike * 2) / 2
round_max_feelslike = round(max_feelslike * 2) / 2
temp_points = np.arange(round_min_feelslike, round_max_feelslike, 0.5)
print(temp_points)

average_drives_temp = np.zeros([len(temp_points),2])
for i in range(len(temp_points)):
    drives_temp = []
    for j in range(data_points):
        if data['roundedfeelslike'][j] == temp_points[i]:
            drives_temp.append(data['Taxi Drives'][j])
    average_drives = np.average(drives_temp)
    average_drives_temp[i] = [temp_points[i], average_drives]
    
print(average_drives_temp)

rounded_feelslikes_2 = np.zeros(data_points)
for i in range(data_points):
    rounded_feelslike = np.round(data['feelslike'][i])
    rounded_feelslikes_2[i] = rounded_feelslike

data['roundedfeelslike_2'] = rounded_feelslikes_2

round_min_feelslike_2 = round(min_feelslike)
round_max_feelslike_2 = round(max_feelslike)
temp_points_2 = np.arange(round_min_feelslike_2, round_max_feelslike_2, 1)

average_drives_temp_2 = np.zeros([len(temp_points_2),2])
for i in range(len(temp_points_2)):
    drives_temp_2 = []
    for j in range(data_points):
        if data['roundedfeelslike_2'][j] == temp_points_2[i]:
            drives_temp_2.append(data['Taxi Drives'][j])
    average_drives_2 = np.average(drives_temp_2)
    average_drives_temp_2[i] = [temp_points_2[i], average_drives_2]
    
print(average_drives_temp_2)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(average_drives_temp[:,0], average_drives_temp[:,1], 'r', label='Taxi Drives')
ax.plot(average_drives_temp_2[:,0], average_drives_temp_2[:,1], 'b', label='Taxi Drives')

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(average_drives_temp[:,0], average_drives_temp[:,1])
ax.scatter(average_drives_temp_2[:,0], average_drives_temp_2[:,1])

print(data['feelslike'])