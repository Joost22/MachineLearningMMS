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
path = os.getcwd() + '\Weather_TaxiDrives.csv'
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