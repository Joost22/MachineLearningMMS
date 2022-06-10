# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:58:23 2022

@author: Mathan Lokerse
"""
import pyarrow.parquet as pq
from sklearn import linear_model
from sklearn.model_selection import train_test_split 
import numpy as np

trips = pq.read_table('yellow_tripdata_2022-01.parquet')
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

# def calculate_error(a,b):
#     error = np.sqrt((a[0]-b[0])**2)

#     return error 

# def assign_centroids(data, centroids):
#     '''
#     Receives a dataframe of data and centroids and returns a list assigning each observation to a centroid.
#     data: a dataframe with all data that will be used.
#     centroids: a dataframe with the centroids. 
#     '''
#     n_observations = data.shape[0]
#     centroid_assign = []
#     centroid_errors = []
#     k = centroids.shape[0]

#     for observation in range(n_observations):

#         # Calculate the error (distance) between each observation and the centroids
#         errors = np.array([])
#         #print('errors=', errors)
#         for centroid in range(k):
#             error = calculate_error(centroids.iloc[centroid, :2], data.iloc[observation,:2])
#             errors = np.append(errors, error)

#         # Calculate closest centroid & error 
#         #!!!  IMPLEMENT the assignment  to the closest centroid !!! #
        
#         closest_centroid = np.argmin(errors)
#         centroid_error = errors[closest_centroid]
        
#         # Assign values to lists
#         centroid_assign.append(closest_centroid)
#         centroid_errors.append(centroid_error)

#     #print(centroid_assign)
#     #print(centroid_errors)    
#     return (centroid_assign,centroid_errors)

# error = []
# WillContinue = True
# i=0
# while(WillContinue):
#     # PHASE 1 - assigns each observation to the nearest centroid
#     # Obtain assigned centroids and the associated error
#     locations['centroid'], iter_error = assign_centroids(locations,centroids)
#     error.append(sum(iter_error))
    
#     #PHASE 2 - updates the cluster centroids based on the assigned observations
#     # Based on the assignment of the observations, recalculate centroids, namely the mean of the observations in the same cluster
#     centroids = locations.groupby('centroid').agg('mean').reset_index(drop = True)

#     # Check if the error has decreased
#     if(len(error)<2):
#         WillContinue = True #we continue if we are still able to reduce
#     else:
#         if(round(error[i],3) !=  round(error[i-1],3)):
#             WillContinue = True
#         else: # if we are not able to improve anymore at all we stop
#             WillContinue = False 
#     i = i + 1 

# #Final centroids together with their error
# locations['centroid'], iter_error = assign_centroids(locations,centroids)
# centroids = locations.groupby('centroid').agg('mean').reset_index(drop = True)

# # from sklearn.cluster import MiniBatchKMeans

# # kmeans = MiniBatchKMeans(n_clusters=15, batch_size=10000, random_state = 42).fit(locations) #fit to 15 clusters using all coordinates
# # locations['label'] = kmeans.labels_

# # trips['pickup_cluster'] = kmeans.predict(trips['PULocationID'])
# # trips['dropoff_cluster'] = kmeans.predict(trips['DOLocationID'])