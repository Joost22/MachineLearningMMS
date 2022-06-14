# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:28:31 2022

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

trips = pd.read_csv('trips_01-2018.csv')

##Importing taxi data (Files have to be in same folder as the code file)
#Path_January_2018 = os.getcwd() + '/yellow_tripdata_2018-01.parquet'

#Convert csv files to dataframe
#Split the datetime column into separate date and time columns
#Remove all pickup dates outtside of the month of interest (there is a lot of incorrect data)
#Taxi_January_2018 = pd.read_parquet(Path_January_2018)
#Taxi_January_2018['Date'] = pd.to_datetime(Taxi_January_2018['tpep_pickup_datetime']).dt.date
#Taxi_January_2018['Time'] = pd.to_datetime(Taxi_January_2018['tpep_pickup_datetime']).dt.time
#Taxi_January_2018 = Taxi_January_2018[pd.to_datetime(Taxi_January_2018['Date']).dt.month == 1]
#Taxi_January_2018 = Taxi_January_2018[pd.to_datetime(Taxi_January_2018['Date']).dt.year == 2018]

#find duration of a trip
trips['tpep_pickup_datetime'] = pd.to_datetime(trips['tpep_pickup_datetime'])
trips['tpep_dropoff_datetime'] = pd.to_datetime(trips['tpep_dropoff_datetime'])
trips['diff'] = trips['tpep_dropoff_datetime']-trips['tpep_pickup_datetime']
trips['duration']=trips['diff'].dt.total_seconds()

# Remove unnecessary data
trips.drop(['VendorID'], axis = 1, inplace=True)
trips.drop(['RatecodeID'], axis = 1, inplace=True)
trips.drop(['store_and_fwd_flag'], axis = 1, inplace=True)
trips.drop(['payment_type'], axis = 1, inplace=True)
trips.drop(['extra'], axis = 1, inplace=True)
trips.drop(['mta_tax'], axis = 1, inplace=True)
trips.drop(['tolls_amount'], axis = 1, inplace=True)
trips.drop(['improvement_surcharge'], axis = 1, inplace=True)
trips.drop(['congestion_surcharge'], axis = 1, inplace=True)
trips.drop(['airport_fee'], axis = 1, inplace=True)

##remove outliers 
trips = trips.drop(trips[trips['trip_distance']<=0].index, axis = 0)
trips = trips.drop(trips[trips['total_amount']<=0].index, axis = 0)
trips = trips.drop(trips[trips['fare_amount']<=0].index, axis = 0)
trips = trips.drop(trips[trips['passenger_count']<=0].index, axis = 0)

# Remove all rows with NaN values in the columns 
trips = trips.dropna()

# Data preparing locations
trips['cluster'] = trips['PULocationID']

zones_Bronx = [3, 18, 20, 31, 32, 46, 47, 51, 58, 59, 60, 69, 78, 81, 94, 119, 126, 136, 147, 159, 167, 168, 169, 174, 182, 183, 184, 185, 199, 200, 208, 212, 213, 220, 235, 240, 241, 242, 247, 248, 250, 254, 259]
zones_Brooklyn = [11, 14, 17, 21, 22, 25, 26, 29, 33, 34, 35, 36, 37, 39, 40, 49, 52, 54, 55, 61, 62, 63, 65, 66, 67, 71, 72, 76, 77, 80, 85, 89, 91, 97, 106, 108, 111, 112, 123,133,149,150,154,155,165,177,178,181,188,189,190,195,210,217,222,225,227,228,255,256,257]
zones_Manhattan = [4,12,13,24,41,42,43,45,48,50,68,74,75,79,87,88,90,100,103,104,105,107,113,114,116,120,125,127,128,137,140,141,142,143,144,148,151,152,153,158,161,162,163,164,166,170,186,194,202,209,211,224,229,230,231,232,233,234,236,237,238,239,243,244,246,249,261,262,263]
zones_Queens = [2,7,8,9,10,15,16,19,27,28,30,38,53,56,57,64,70,73,82,83,86,92,93,95,96,98,101,102,117,121,122,124,129,130,131,132,134,135,138,139,145,146,157,160,171,173,175,179,180,191,192,193,196,197,198,201,203,205,207,215,216,218,219,223,226,252,253,258,260]
zones_StatenIsland = [5,6,23,44,84,99,109,110,115,118,156,172,176,187,204,206,214,221,245,251]

for index in trips['PULocationID']:
    if index in zones_Bronx:
        trips['cluster'] = 1
    elif index in zones_Brooklyn:
        trips['cluster'] = 2
    elif index in zones_Manhattan:
        trips['cluster'] = 3
    elif index in zones_Queens:
        trips['cluster'] = 4
    elif index in zones_StatenIsland:
        trips['cluster'] = 5
    else:
        trips['cluster'] = 0

trips = trips.drop(trips[trips['cluster']==0].index, axis = 0)

# data preparing weekend / week day. Monday 0, Tuesday 1
# trips['cluster'] = trips['PULocationID']  
trips['weekday'] = trips['date_time'].dt.dayofweek

#trips['passenger_count'].describe()
#trips['passenger_count'].value_counts()
#trips['duration'].describe()

#trips = trips.drop(trips[trips['duration']<30].index, axis = 0)
#trips = trips.drop(trips[trips['duration']>10000].index, axis = 0)
#trips = trips.drop(trips[trips['passenger_count']<10].index, axis = 0)
#trips = trips.drop(trips[trips['passenger_count']==float("NaN")].index, axis = 0)

fig, ax = plt.subplots(figsize=(12,8))
ax.set_xlabel ('Tip amount')
ax.set_ylabel ('Frequency')
plt.hist(trips['tip_amount'])
plt.legend()
plt.show()  
 
fig, ax = plt.subplots(figsize=(12,8))
ax.set_xlabel ('Weekday')
ax.set_ylabel ('Frequency')
plt.hist(trips['weekday'])
plt.legend()
plt.show() 

fig, ax = plt.subplots(figsize=(12,8))
ax.set_xlabel ('Passenger count')
ax.set_ylabel ('Frequency')
plt.hist(trips['passenger_count'])
plt.legend()
plt.show() 

fig, ax = plt.subplots(figsize=(12,8))
ax.set_xlabel ('Tip amount')
ax.scatter(trips['passenger_count'], trips['trip_amount'])
plt.legend()
plt.show()  







