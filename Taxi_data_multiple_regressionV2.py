# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:55:41 2022

@author: daanv
"""

import pyarrow.parquet as pq
from sklearn import linear_model
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
# plt.rcParams['agg.path.chunksize'] = 10000

trips = pq.read_table('yellow_tripdata_2018-03.parquet')
trips = trips.to_pandas()
trips = trips.sample(n=100000)
#find duration of a trip
trips['diff'] = trips['tpep_dropoff_datetime']-trips['tpep_pickup_datetime']
trips['duration']=trips['diff'].dt.total_seconds()

#find the hour and day of a tri (integer)
trips['tpep_pickup_datetime_day']=trips['tpep_pickup_datetime'].dt.day_of_week
trips['tpep_pickup_datetime_hour']=trips['tpep_pickup_datetime'].dt.hour

#remove outliers 
trips = trips.dropna(axis=0,subset = ['passenger_count'])
trips = trips.drop(trips[trips['duration']<30].index, axis = 0)
trips = trips.drop(trips[trips['duration']>10000].index, axis = 0)
trips = trips.drop(trips[trips['passenger_count']==0].index, axis = 0)
trips = trips.drop(trips[trips['trip_distance']==0].index, axis = 0)
trips = trips.drop(trips[trips['total_amount']==0].index, axis = 0)

# trips = trips.drop(trips[trips['tip_amount']==0].index, axis = 0)
trips = trips.drop(trips[trips['tip_amount']>75].index, axis = 0)

X = trips[['duration']]
y = trips['tip_amount']


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.20, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
prediction = regr.predict(X_test)

print(prediction)

print(regr.score(X_test,Y_test))

fig, ax = plt.subplots(figsize=(12,8))

ax.scatter(X_test['duration'],Y_test)
ax.plot(X_test['duration'],prediction,'r')

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data_new.iloc[:,[1]], data_new.iloc[:,[2]], label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('Taxi Drives')
# ax.set_ylabel('Temperature')
# ax.set_title('Predicted Temperature vs. Taxi Drives')
