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

#Importing taxi data (Files have to be in same folder as the code file)
Path_January_2018 = os.getcwd() + '/yellow_tripdata_2018-01.parquet'
Path_February_2018 = os.getcwd() + '/yellow_tripdata_2018-02.parquet'
#Path_March_2018 = os.getcwd() + '/yellow_tripdata_2018-03.parquet'
#Path_April_2018 = os.getcwd() + '/yellow_tripdata_2018-04.parquet'
#Path_May_2018 = os.getcwd() + '/yellow_tripdata_2018-05.parquet'
#Path_June_2018 = os.getcwd() + '/yellow_tripdata_2018-06.parquet'
#Path_July_2018 = os.getcwd() + '/yellow_tripdata_2018-07.parquet'
#Path_August_2018 = os.getcwd() + '/yellow_tripdata_2018-08.parquet'
#Path_September_2018 = os.getcwd() + '/yellow_tripdata_2018-09.parquet'
#Path_October_2018 = os.getcwd() + '/yellow_tripdata_2018-10.parquet'
#Path_November_2018 = os.getcwd() + '/yellow_tripdata_2018-11.parquet'
#Path_December_2018 = os.getcwd() + '/yellow_tripdata_2018-12.parquet'

#Convert csv files to dataframe
#Split the datetime column into separate date and time columns
#Remove all pickup dates outtside of the month of interest (there is a lot of incorrect data)
Taxi_January_2018 = pd.read_parquet(Path_January_2018)
Taxi_January_2018['Date'] = pd.to_datetime(Taxi_January_2018['tpep_pickup_datetime']).dt.date
Taxi_January_2018['Time'] = pd.to_datetime(Taxi_January_2018['tpep_pickup_datetime']).dt.time
Taxi_January_2018 = Taxi_January_2018[pd.to_datetime(Taxi_January_2018['Date']).dt.month == 1]
Taxi_January_2018 = Taxi_January_2018[pd.to_datetime(Taxi_January_2018['Date']).dt.year == 2018]

Taxi_February_2018 = pd.read_parquet(Path_February_2018)
Taxi_February_2018['Date'] = pd.to_datetime(Taxi_February_2018['tpep_pickup_datetime']).dt.date
Taxi_February_2018['Time'] = pd.to_datetime(Taxi_February_2018['tpep_pickup_datetime']).dt.time
Taxi_February_2018 = Taxi_February_2018[pd.to_datetime(Taxi_February_2018['Date']).dt.month == 2]
Taxi_February_2018 = Taxi_February_2018[pd.to_datetime(Taxi_February_2018['Date']).dt.year == 2018]

#Taxi_March_2018 = pd.read_parquet(Path_March_2018)
#Taxi_March_2018['Date'] = pd.to_datetime(Taxi_March_2018['tpep_pickup_datetime']).dt.date
#Taxi_March_2018['Time'] = pd.to_datetime(Taxi_March_2018['tpep_pickup_datetime']).dt.time
#Taxi_March_2018 = Taxi_March_2018[pd.to_datetime(Taxi_March_2018['Date']).dt.month == 3]
#Taxi_March_2018 = Taxi_March_2018[pd.to_datetime(Taxi_March_2018['Date']).dt.year == 2018]
#
#Taxi_April_2018 = pd.read_parquet(Path_April_2018)
#Taxi_April_2018['Date'] = pd.to_datetime(Taxi_April_2018['tpep_pickup_datetime']).dt.date
#Taxi_April_2018['Time'] = pd.to_datetime(Taxi_April_2018['tpep_pickup_datetime']).dt.time
#Taxi_April_2018 = Taxi_April_2018[pd.to_datetime(Taxi_April_2018['Date']).dt.month == 4]
#Taxi_April_2018 = Taxi_April_2018[pd.to_datetime(Taxi_April_2018['Date']).dt.year == 2018]
#
#Taxi_May_2018 = pd.read_parquet(Path_May_2018)
#Taxi_May_2018['Date'] = pd.to_datetime(Taxi_May_2018['tpep_pickup_datetime']).dt.date
#Taxi_May_2018['Time'] = pd.to_datetime(Taxi_May_2018['tpep_pickup_datetime']).dt.time
#Taxi_May_2018 = Taxi_May_2018[pd.to_datetime(Taxi_May_2018['Date']).dt.month == 5]
#Taxi_May_2018 = Taxi_May_2018[pd.to_datetime(Taxi_May_2018['Date']).dt.year == 2018]
#
#Taxi_June_2018 = pd.read_parquet(Path_June_2018)
#Taxi_June_2018['Date'] = pd.to_datetime(Taxi_June_2018['tpep_pickup_datetime']).dt.date
#Taxi_June_2018['Time'] = pd.to_datetime(Taxi_June_2018['tpep_pickup_datetime']).dt.time
#Taxi_June_2018 = Taxi_June_2018[pd.to_datetime(Taxi_June_2018['Date']).dt.month == 6]
#Taxi_June_2018 = Taxi_June_2018[pd.to_datetime(Taxi_June_2018['Date']).dt.year == 2018]
#
#Taxi_July_2018 = pd.read_parquet(Path_July_2018)
#Taxi_July_2018['Date'] = pd.to_datetime(Taxi_July_2018['tpep_pickup_datetime']).dt.date
#Taxi_July_2018['Time'] = pd.to_datetime(Taxi_July_2018['tpep_pickup_datetime']).dt.time
#Taxi_July_2018 = Taxi_July_2018[pd.to_datetime(Taxi_July_2018['Date']).dt.month == 7]
#Taxi_July_2018 = Taxi_July_2018[pd.to_datetime(Taxi_July_2018['Date']).dt.year == 2018]
#
#Taxi_August_2018 = pd.read_parquet(Path_August_2018)
#Taxi_August_2018['Date'] = pd.to_datetime(Taxi_August_2018['tpep_pickup_datetime']).dt.date
#Taxi_August_2018['Time'] = pd.to_datetime(Taxi_August_2018['tpep_pickup_datetime']).dt.time
#Taxi_August_2018 = Taxi_August_2018[pd.to_datetime(Taxi_August_2018['Date']).dt.month == 8]
#Taxi_August_2018 = Taxi_August_2018[pd.to_datetime(Taxi_August_2018['Date']).dt.year == 2018]
#
#Taxi_September_2018 = pd.read_parquet(Path_September_2018)
#Taxi_September_2018['Date'] = pd.to_datetime(Taxi_September_2018['tpep_pickup_datetime']).dt.date
#Taxi_September_2018['Time'] = pd.to_datetime(Taxi_September_2018['tpep_pickup_datetime']).dt.time
#Taxi_September_2018 = Taxi_September_2018[pd.to_datetime(Taxi_September_2018['Date']).dt.month == 9]
#Taxi_September_2018 = Taxi_September_2018[pd.to_datetime(Taxi_September_2018['Date']).dt.year == 2018]
#
#Taxi_October_2018 = pd.read_parquet(Path_October_2018)
#Taxi_October_2018['Date'] = pd.to_datetime(Taxi_October_2018['tpep_pickup_datetime']).dt.date
#Taxi_October_2018['Time'] = pd.to_datetime(Taxi_October_2018['tpep_pickup_datetime']).dt.time
#Taxi_October_2018 = Taxi_October_2018[pd.to_datetime(Taxi_October_2018['Date']).dt.month == 10]
#Taxi_October_2018 = Taxi_October_2018[pd.to_datetime(Taxi_October_2018['Date']).dt.year == 2018]
#
#Taxi_November_2018 = pd.read_parquet(Path_November_2018)
#Taxi_November_2018['Date'] = pd.to_datetime(Taxi_November_2018['tpep_pickup_datetime']).dt.date
#Taxi_November_2018['Time'] = pd.to_datetime(Taxi_November_2018['tpep_pickup_datetime']).dt.time
#Taxi_November_2018 = Taxi_November_2018[pd.to_datetime(Taxi_November_2018['Date']).dt.month == 11]
#Taxi_November_2018 = Taxi_November_2018[pd.to_datetime(Taxi_November_2018['Date']).dt.year == 2018]
#
#Taxi_December_2018 = pd.read_parquet(Path_December_2018)
#Taxi_December_2018['Date'] = pd.to_datetime(Taxi_December_2018['tpep_pickup_datetime']).dt.date
#Taxi_December_2018['Time'] = pd.to_datetime(Taxi_December_2018['tpep_pickup_datetime']).dt.time
#Taxi_December_2018 = Taxi_December_2018[pd.to_datetime(Taxi_December_2018['Date']).dt.month == 12]
#Taxi_December_2018 = Taxi_December_2018[pd.to_datetime(Taxi_December_2018['Date']).dt.year == 2018]

trips = pd.concat([Taxi_January_2018 , Taxi_February_2018])#, Taxi_March_2018, Taxi_April_2018, Taxi_May_2018, Taxi_June_2018, Taxi_July_2018, Taxi_August_2018, Taxi_September_2018, Taxi_October_2018, Taxi_November_2018, Taxi_December_2018], axis=0)
trips.shape

#find duration of a trip
trips['diff'] = trips['tpep_dropoff_datetime']-trips['tpep_pickup_datetime']
trips['duration']=trips['diff'].dt.total_seconds()

#remove outliers 
trips = trips.drop(trips[trips['duration']<30].index, axis = 0)
trips = trips.drop(trips[trips['duration']>10000].index, axis = 0)
trips = trips.drop(trips[trips['passenger_count']<=0].index, axis = 0)
trips = trips.drop(trips[trips['passenger_count']<10].index, axis = 0)
trips = trips.drop(trips[trips['passenger_count']==float("NaN")].index, axis = 0)
trips = trips.drop(trips[trips['trip_distance']<=0].index, axis = 0)
trips = trips.drop(trips[trips['total_amount']<=0].index, axis = 0)
trips = trips.drop(trips[trips['fare_amount']<=0].index, axis = 0)






