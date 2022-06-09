# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:55:41 2022

@author: daanv
"""

import pyarrow.parquet as pq
from sklearn import linear_model
from sklearn.model_selection import train_test_split 

trips = pq.read_table('yellow_tripdata_2018-03.parquet')
trips = trips.to_pandas()

#find duration of a trip
trips['diff'] = trips['tpep_dropoff_datetime']-trips['tpep_pickup_datetime']
trips['duration']=trips['diff'].dt.total_seconds()

#find the hour and day of a tri (integer)
trips['tpep_pickup_datetime_day']=trips['tpep_pickup_datetime'].dt.day_of_week
trips['tpep_pickup_datetime_hour']=trips['tpep_pickup_datetime'].dt.hour

#remove outliers 
trips = trips.drop(trips[trips['duration']<30].index, axis = 0)
trips = trips.drop(trips[trips['duration']>10000].index, axis = 0)
trips = trips.drop(trips[trips['passenger_count']==0].index, axis = 0)
trips = trips.drop(trips[trips['trip_distance']==0].index, axis = 0)
trips = trips.drop(trips[trips['total_amount']==0].index, axis = 0)

X = trips[['duration','trip_distance']]
y = trips['tip_amount']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.20, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
predicted_taxi_amount = regr.predict(X_test)

print(predicted_taxi_amount)

print(regr.score(X_test,Y_test))

