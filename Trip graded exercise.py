# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:28:31 2022

@author: roosm
"""

# Import relevant packages
import time
start_time = time.time()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


### --- DATA PREPARATION ---

# Import relevant packages
import pyarrow.parquet as pq

# Read the .parquet data as dataframe
alltripdata = pq.read_table('yellow_tripdata_2018-01.parquet')
alltripdata = alltripdata.to_pandas()

# Set datetime columns to the correct .pd format
alltripdata['tpep_pickup_datetime'] = pd.to_datetime(alltripdata['tpep_pickup_datetime'])
alltripdata['tpep_dropoff_datetime'] = pd.to_datetime(alltripdata['tpep_dropoff_datetime'])

# Find the day and hour of the trips
alltripdata['trip_day'] = alltripdata['tpep_pickup_datetime'].dt.day
alltripdata['trip_hour'] = alltripdata['tpep_pickup_datetime'].dt.hour

# Calculate the trip durations in seconds
alltripdata['trip_duration'] = (alltripdata['tpep_dropoff_datetime'] - alltripdata['tpep_pickup_datetime']).dt.total_seconds()

# Calculate the tips as percentage of the fare
alltripdata['tip_percentage'] = alltripdata['tip_amount'] / alltripdata['fare_amount']


### --- DATA FILTERING ---

# Remove all trips that aren't paid by creditcard (= 1), these tips are unreliable since not all cash tips are declared due to taxation
alltripdata = alltripdata[alltripdata.payment_type == 1]

# Remove all trips with 0 or NaN column values except for the 'trip_hour', tip_amount' columns since these can be 0.
alltripdata = alltripdata.replace(0, np.nan).dropna(axis=0, how='any', subset = ['trip_duration', 'payment_type', 'fare_amount', 'PULocationID', 'DOLocationID', 'trip_distance'])
alltripdata = alltripdata.fillna(0)

# Remove all negative tips
alltripdata = alltripdata[alltripdata.tip_amount >= 0]

# Remove all tips over 100$ (assuming that tips above 100$ are errors)
alltripdata = alltripdata[alltripdata.tip_amount <= 100]

# Remove trips with duration below 30 seconds
alltripdata = alltripdata[alltripdata.trip_duration >= 30]

# Remove trips with duration above 10800 seconds (3 hours)
alltripdata = alltripdata[alltripdata.trip_duration <= 10800]

# Remove trips with trip_distance above 100 miles (probably errors for intra-city trips)
alltripdata = alltripdata[alltripdata.trip_distance <= 100]

# Remove trips with trip_distance under 0.1 miles
alltripdata = alltripdata[alltripdata.trip_distance >= 0.1]

# Remove trips with fare_amount above $250 
alltripdata = alltripdata[alltripdata.fare_amount <= 250]

# Remove trips with fare_amount under $2.50 initial charge
alltripdata = alltripdata[alltripdata.fare_amount >= 2.5]

# Remove trips with ratecodeID's of 99
alltripdata = alltripdata[alltripdata.RatecodeID <= 6]

# Remove trips with passengercount higher than 6
alltripdata = alltripdata[alltripdata.passenger_count <= 6]

# Remove all tips over 500% of fare (assuming that tips above 500% are errors)
alltripdata =alltripdata[alltripdata.tip_percentage <= 5]

print(alltripdata)

# Check correlation, this gives an idea of the relation between different variables/features
fig, axes = plt.subplots(figsize = (12,6))
correlation = alltripdata.corr()
corr_m = sns.heatmap(round(correlation, 2), annot=True, cmap='Blues', ax=axes, fmt='.2f')

# Extract the relevant columns
tripdata = alltripdata[['RatecodeID', 'trip_duration', 'fare_amount', 'payment_type', 'PULocationID', 'DOLocationID', 'trip_distance', 'tip_amount']]

# Check correlation, this gives an idea of the relation between different variables/features
fig, axes = plt.subplots(figsize = (12, 6))
correlation = tripdata.corr()
corr_m = sns.heatmap(round(correlation, 2), annot=True, cmap='Blues', ax=axes, fmt='.2f')

### --- VISUALISE POSSIBLE RELATIONSHIPS ---

#distribution of tip_amount (dollars)
fig, ax = plt.subplots(figsize=(12,8))
plt.hist(tripdata['tip_amount'], 500, color='pink', alpha=0.5)
plt.title('Tip amount histogram')
plt.xlabel('Tip amount')
plt.xlim(0, 20)
plt.ylabel('Frequency')
plt.show()

# # tip_percentage and fare_amount (dollars)
# plt.scatter(tripdata['fare_amount'], tripdata['tip_percentage'], c='yellow', alpha=0.5)
# plt.title('Fare amount vs. taxi tips')
# plt.xlabel('Fare amount (dollars)')
# plt.ylabel('Tip as percentage of fare')
# plt.show()

# # tip_percentage and trip_distance (miles)
# plt.scatter(tripdata['trip_distance'], tripdata['tip_percentage'], c='red', alpha=0.5)
# plt.title('Trip distance vs. taxi tips')
# pplt..xlabel('Trip distance (miles)')
# plt.ylabel('Tip as percentage of fare')
# plt.show()

# # tip_percentage and trip_duration
# plt.scatter(tripdata['trip_duration'], tripdata['tip_percentage'], c='blue', alpha=0.5)
# plt.title('Trip duration vs. taxi tips')
# plt.xlabel('Trip duration (seconds)')
# plt.ylabel('Tip as percentage of fare')
# plt.show()

# # tip_percentage and passenger_count
# plt.scatter(tripdata['passenger_count'], tripdata['tip_percentage'], c='green', alpha=0.5)
# plt.title('Passenger count vs. taxi tips')
# plt.xlabel('Passenger count')
# plt.ylabel('Tip as percentage of fare')
# plt.show()

# # tip_percentage and hour of day
# plt.scatter(tripdata['trip_hour'], tripdata['tip_percentage'], c='magenta', alpha=0.5)
# plt.title('Hour of day vs. taxi tips')
# plt.xlabel('Hour of day')
# plt.ylabel('Tip as percentage of fare')
# plt.show()


### --- FURTHER DATA PREPARATION ---

# Add an empty column for the tip classes
tripdata['tip_class'] = " "

# Assign tip classes: 'no tip', 'regular' and 'generous' (more than 30% of fare)
tripdata.loc[tripdata['tip_amount'] == 0, 'tip_class'] = 'no tip'
tripdata.loc[(tripdata['tip_amount'] > 0) & (tripdata['tip_amount'] <= 4), 'tip_class'] = 'regular'
tripdata.loc[tripdata['tip_amount'] > 4, 'tip_class'] = 'generous'

# Print the occurrence of the different tip classes
print('--- Unbalanced tip class occurrence: ---\n',tripdata['tip_class'].value_counts())

# From the printed data it becomes visible that tip classes do not occur in equal shares
# This means the data needs to be balanced. 'No tip' is the limiting class due to its most limited occurrence.
notip = tripdata.loc[tripdata['tip_class'] == 'no tip']

# We randomly select an equal amount of trips from the 'regular' and 'generous' class
regtip = tripdata.loc[tripdata['tip_class'] == 'regular']
regtip = regtip.sample(n=len(notip))

gentip = tripdata.loc[tripdata['tip_class'] == 'generous']
gentip = gentip.sample(n=len(notip))

# Now we append the balanced classes in a single dataframe where the new shares are (1:1:1)
balancedtripdata = notip.append([regtip,gentip])

## Check if the balancing worked
# Visualise the occurrence of the different tip classes
print('--- Balanced tip class occurrence: ---\n',balancedtripdata['tip_class'].value_counts())


## --- FINAL DATA PREPARATION FOR MLR AND MLP CLASSIFIER ---

# Setting the features (X) and dependent variable (Y)
X = balancedtripdata.iloc[:,0:7]
Y = balancedtripdata['tip_class']

# Label encoding is required since the model only accepts numeric values
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

LE = preprocessing.LabelEncoder()
LE.fit(Y)
Y = LE.transform(Y)

# In order to know which class has been assigned which label
LE_name_mapping = dict(zip(LE.classes_, LE.transform(LE.classes_)))
print('Classes are label encoded as follows: \n',LE_name_mapping)


# Split the data into training, validation and test data (60:20:20)
from sklearn.model_selection import train_test_split

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

# ### --- MULTINOMIAL LOGISTIC REGRESSION ---

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# # Train the MLR model
# train_logreg = LogisticRegression(random_state=1,max_iter = 300)
# train_logreg.fit(X_train_scale,Y_train)

# # Prediction with MLR to the train data
# pred_logreg = train_logreg.predict(X_train_scale)
# print("For Logistic Regression: ")
# print(classification_report(Y_train, pred_logreg))
# print ("Accuracy of the above model is: ",accuracy_score(pred_logreg,Y_train))

# # Prediction with MLR to the validation data
# pred_logreg = train_logreg.predict(X_val_scale)
# print("For Logistic Regression: ")
# print(classification_report(Y_val, pred_logreg))
# print ("Accuracy of the above model is: ",accuracy_score(pred_logreg,Y_val))

# # Evaluate model with cross-validation on the test data
# from sklearn.model_selection import cross_val_score
# cross_val = cross_val_score(train_logreg, X_test_scale ,Y_test)#, cv=10)
# print("%0.2f accuracy with a standard deviation of %0.2f" % (cross_val.mean(), cross_val.std()))
# cross_val = pd.Series(cross_val)
# print((cross_val.min(), cross_val.mean(), cross_val.max()))

# ### --- MULTI-LAYER PERCEPTRON CLASSIFIER ---

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

# Train the neural network
train_nn = MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(128,64,32,16,), random_state=1, verbose=True)
train_nn.fit(X_train_scale,Y_train)

# Prediction with Multi-layer Perceptron Classifier
pred_nn = train_nn.predict(X_val_scale)
print("For Neural Network: ")
print(classification_report(Y_val, pred_nn))
print ("Accuracy of the above model is: ",accuracy_score(pred_nn,Y_val))

print(confusion_matrix(Y_val, pred_nn))

### --- TIMEKEEPING ---

print("--- %s seconds to run the code ---" % (time.time() - start_time))