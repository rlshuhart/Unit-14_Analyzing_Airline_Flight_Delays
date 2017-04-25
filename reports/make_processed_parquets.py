import dask.dataframe as dd #http://dask.pydata.org/en/latest/
import pandas as pd
import numpy as np
from datetime import datetime

# Data Location
parq_folder = "../data/parquet-tiny/" # Testing
#parq_folder = "../data/parquet_25/" # Higher Load Testing
#parq_folder = "../data/parquet/" # Full data
print(parq_folder)
print(datetime.now())
df = dd.read_parquet(parq_folder)

# Create an hour field
# 2400 minutes from midnight reduced to 2399 then int division drops to 23
df = df.assign(Hour=df.CRSDepTime.astype(float).clip(upper=2399)//100)

# Months from 0 AD
df['FlightAge'] = 12*df['Year']+df['Month']-1

# The months from the first recorded flight is consider the approx age of the plane.
# Unfortunately, tail numbers not tracked until 1995.

# Find the first year and month of a tail numbers flight history
tail_births = (df.groupby('TailNum')[['FlightAge']].min().reset_index()
                 .rename(columns={'FlightAge':'FirstFlight'}))

df_with_tails = dd.merge(df[df['Year']>1994], tail_births, how='left', on='TailNum')
df_with_tails['Age'] = df_with_tails['FlightAge'] - df_with_tails['FirstFlight']

#df_with_tails = df_with_tails.drop(['FlightAge','FirstFlight'], axis=1)

start = datetime.now()
def scaler(df, column):
    return (df[column] - df[column].dropna().mean())/df[column].dropna().std()

# Scale columns for regression of all data
df['Hour_scaled'] = scaler(df, 'Hour')
df['Distance_scaled'] = scaler(df, 'Distance')

# Scale columns for regression for after 1994
df_with_tails['Hour_scaled'] = scaler(df_with_tails, 'Hour')
df_with_tails['Distance_scaled'] = scaler(df_with_tails, 'Distance')
df_with_tails['Age_scaled'] = scaler(df_with_tails, 'Age')

print("Time to Build: ", datetime.now() - start)

start = datetime.now()
print("Starting all processed...")
df.to_parquet(parq_folder+"processed/", compression='gzip', object_encoding='utf8')
print("Time to make processed data: ", datetime.now() - start)

print("Starting with plane ages...")
df_with_tails.to_parquet(parq_folder+"processed_age/", compression='gzip', object_encoding='utf8')
print("Time to make processed data with plane ages: ", datetime.now() - start)
