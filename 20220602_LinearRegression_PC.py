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
path = os.getcwd() + 'Weather_TaxiDrives.csv'
data = pd.read_csv(path, header=0)
data.head()

data.describe()

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))