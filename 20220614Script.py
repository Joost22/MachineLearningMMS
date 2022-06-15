#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 12:48:00 2022

@author: PieterClemens
"""
# Import relevant packages

import numpy as np
import pandas as pd
import seaborn as sns
import os


### --- DATA PREPARATION ---

# Import relevant packages
import pyarrow.parquet as pq

# Read the .parquet data as dataframe
alltrips = pq.read_table('yellow_tripdata_2022-01.parquet')
alltrips = alltrips.to_pandas()

print(alltrips)

# Extract the relevant columns from the data
trips = alltrips[['tpep_pickup_datetime','tpep_dropoff_datetime','passenger_count','trip_distance','PULocationID','DOLocationID','tip_amount']]

# Remove all rows with zeroes and NaN values in the columns (except for the tip column, since tips can be 0$)
trips = trips.replace(0, np.nan).dropna(axis=0, how='any', subset = ['tpep_pickup_datetime','tpep_dropoff_datetime','passenger_count','trip_distance','PULocationID','DOLocationID'])
trips = trips.fillna(0)

# Remove all rows with negative tips (assuming tips below 0$ are errors)
negtips = trips[trips['tip_amount'] < 0].index
trips.drop(negtips, inplace=True)

# Remove all rows with too high tips (assuming tips above 100$ are errors)
hightips = trips[trips['tip_amount'] > 100].index
trips.drop(hightips, inplace=True)

print(trips)

# Determine tip mean, median, standard deviation, maximum and minimum
tipmean = trips['tip_amount'].mean()
tipstd = trips['tip_amount'].std()
tipmax = trips['tip_amount'].max()
tipmin = trips['tip_amount'].min()
tipmedian = trips['tip_amount'].quantile(q=0.62)

print(trips['tip_amount'].describe())

# Add an empty column for the tip classes
trips['tip_class'] = " "

# Assign 7 tip classes; 'no tip', 'low' (below average, not 0), 'average' (average ± 1$), 'high' (above average, not top 1%), 'very high' (top 1%)
trips.loc[trips['tip_amount'] == 0, 'tip_class'] = 'no tip'
trips.loc[(trips['tip_amount'] > 0) & (trips['tip_amount'] < tipmean - 1), 'tip_class'] = 'low'
trips.loc[(trips['tip_amount'] >= tipmean - 1) & (trips['tip_amount'] <= tipmean + 1), 'tip_class'] = 'average'
trips.loc[(trips['tip_amount'] > tipmean + 1) & (trips['tip_amount'] <= trips['tip_amount'].quantile(q=0.99)), 'tip_class'] = 'high'
trips.loc[trips['tip_amount'] > trips['tip_amount'].quantile(q=0.99), 'tip_class'] = 'very high'

ax = trips['tip_class'].value_counts().plot(kind='bar', figsize=(14,8), title="Occurrence per Tip Class")
ax.set_xlabel("Tip Class")
ax.set_ylabel("Frequency")


### --- DATA PREPARATION FOR THE NEURAL NETWORK ---

# Import relevant packages
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report

# Setting the features (X) and dependent variables (Y)
X = trips.iloc[:,0:6]
Y = trips['tip_class']

# Label encoding is required since the model only accepts numeric values
LE = preprocessing.LabelEncoder()
LE.fit(Y)
Y = LE.transform(Y)

objList = X.select_dtypes(include = "datetime64[ns]").columns

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for feat in objList:
    X[feat] = LE.fit_transform(X[feat].astype(str))

# Split the data into training, validation and test data (60:20:20)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 1)
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.25,random_state = 1)

# Datapoints also need to be scaled into dataset with mean 0 and std = 1
X_train_scale = preprocessing.scale(X_train)
X_val_scale = preprocessing.scale(X_val)
X_test_scale = preprocessing.scale(X_test)

# Print the number of data points in training, validation, and test dataset.
print("Datapoints in training set:",len(X_train))
print("Datapoints in validation set:",len(X_val))
print("Datapoints in test set:",len(X_test))


### --- TRAINING THE NEURAL NETWORK ---

# Import relevant packages
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

train_logreg = LogisticRegression(random_state=0,max_iter = 300).fit(X_train_scale,Y_train)

train_nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300,activation = 'relu',solver='adam',random_state=1)

train_nn.fit(X_train,Y_train)

pred_logreg = train_logreg.predict(X_val_scale)
print("For Logistic Regression: ")
print(classification_report(Y_val, pred_logreg))
print ("Accuracy of the above model is: ",accuracy_score(pred_logreg,Y_val))

pred_nn = train_nn.predict(X_val)
print("For Neural Network: ")
print(classification_report(Y_val, pred_nn))
print ("Accuracy of the above model is: ",accuracy_score(pred_nn,Y_val))



















