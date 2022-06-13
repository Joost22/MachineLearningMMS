#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 12:48:00 2022

@author: PieterClemens
"""
# Import relevant libraries

import numpy as np
import pandas as pd
import seaborn as sns
import os


### --- DATA PREPARATION ---

# Read the .parquet data as dataframe
import pyarrow.parquet as pq
alltrips = pq.read_table('yellow_tripdata_2022-01.parquet')
alltrips = alltrips.to_pandas()

print(alltrips)

# Extract the relevant columns from the data
trips = alltrips[['tpep_pickup_datetime','passenger_count','trip_distance','PULocationID','DOLocationID','tip_amount']]

# Remove all rows with zeroes and NaN values in the columns (except for the tip column, since tips can be 0$)
trips = trips.replace(0, np.nan).dropna(axis=0, how='any', subset = ['tpep_pickup_datetime','passenger_count','trip_distance','PULocationID','DOLocationID'])
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
tipmedian = trips['tip_amount'].quantile(q=0.5)

print(trips['tip_amount'].describe())

# Add an empty column for the tip classes
trips['tip_class'] = " "

# Assign 6 tip classes; 'no tip', 'below average' (lowest 25%), 'average' (between 25% and 50%), 'high' (upper 5%), 'super high' (upper 1%)
trips.loc[trips['tip_amount'] == 0, 'tip_class'] = 'no tip'
trips.loc[(trips['tip_amount'] < trips['tip_amount'].quantile(q=0.25)) & (trips['tip_amount'] != 0), 'tip_class'] = 'below average'
trips.loc[(trips['tip_amount'] >= trips['tip_amount'].quantile(q=0.25)) & (trips['tip_amount'] <= trips['tip_amount'].quantile(q=0.75)), 'tip_class'] = 'average'
trips.loc[(trips['tip_amount'] > trips['tip_amount'].quantile(q=0.75)) & (trips['tip_amount'] <= trips['tip_amount'].quantile(q=0.95)), 'tip_class'] = 'high'
trips.loc[(trips['tip_amount'] > trips['tip_amount'].quantile(q=0.95)) & (trips['tip_amount'] <= trips['tip_amount'].quantile(q=0.99)), 'tip_class'] = 'very high'
trips.loc[trips['tip_amount'] > trips['tip_amount'].quantile(q=0.99), 'tip_class'] = 'super high'

sns.displot(trips['tip_class']) # misschien moet alles 20% zijn?

### --- GETTING STARTED FOR THE NEURAL NETWORK ---


## Save the prepared data as .csv

# from pathlib import Path
# filepath = Path('/Users/PieterClemens/Documents/20220612MachineLearning/Output/Trips.csv')
# filepath.parent.mkdir(parents = True, exist_ok = True)
# trips.to_csv(filepath)