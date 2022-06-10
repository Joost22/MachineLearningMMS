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

locations = trips[['PULocationID', 'DOLocationID']]
centroids = locations.sample(15)

def calculate_error(a,b):
    error = np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    return error 

def assign_centroids(data, centroids):
    '''
    Receives a dataframe of data and centroids and returns a list assigning each observation to a centroid.
    data: a dataframe with all data that will be used.
    centroids: a dataframe with the centroids. 
    '''
    n_observations = data.shape[0]
    centroid_assign = []
    centroid_errors = []
    k = centroids.shape[0]

    for observation in range(n_observations):

        # Calculate the error (distance) between each observation and the centroids
        errors = np.array([])
        #print('errors=', errors)
        for centroid in range(k):
            error = calculate_error(centroids.iloc[centroid, :2], data.iloc[observation,:2])
            errors = np.append(errors, error)

        # Calculate closest centroid & error 
        #!!!  IMPLEMENT the assignment  to the closest centroid !!! #
        
        closest_centroid = np.argmin(errors)
        centroid_error = errors[closest_centroid]
        
        # Assign values to lists
        centroid_assign.append(closest_centroid)
        centroid_errors.append(centroid_error)

    #print(centroid_assign)
    #print(centroid_errors)    
    return (centroid_assign,centroid_errors)

error = []
WillContinue = True
i=0
while(WillContinue):
    # PHASE 1 - assigns each observation to the nearest centroid
    # Obtain assigned centroids and the associated error
    locations['centroid'], iter_error = assign_centroids(locations,centroids)
    error.append(sum(iter_error))
    
    #PHASE 2 - updates the cluster centroids based on the assigned observations
    # Based on the assignment of the observations, recalculate centroids, namely the mean of the observations in the same cluster
    centroids = locations.groupby('centroid').agg('mean').reset_index(drop = True)

    # Check if the error has decreased
    if(len(error)<2):
        WillContinue = True #we continue if we are still able to reduce
    else:
        if(round(error[i],3) !=  round(error[i-1],3)):
            WillContinue = True
        else: # if we are not able to improve anymore at all we stop
            WillContinue = False 
    i = i + 1 

#Final centroids together with their error
locations['centroid'], iter_error = assign_centroids(locations,centroids)
centroids = locations.groupby('centroid').agg('mean').reset_index(drop = True)

# from sklearn.cluster import MiniBatchKMeans

# kmeans = MiniBatchKMeans(n_clusters=15, batch_size=10000, random_state = 42).fit(locations) #fit to 15 clusters using all coordinates
# locations['label'] = kmeans.labels_

# trips['pickup_cluster'] = kmeans.predict(trips['PULocationID'])
# trips['dropoff_cluster'] = kmeans.predict(trips['DOLocationID'])