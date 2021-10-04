# https://www.youtube.com/watch?v=FzMQs8oTS3Q
# Trial based on Youtube guide

import glob
from netCDF4 import Dataset 
import pandas as pd
import numpy as np

# Record all the years of the netCDF files into a Python list
all_years = []

for file in glob.glob('*.nc'):
    print(file)
    data = Dataset(file, 'r')
    time = data.variables['time']
    year = time.units[14:18]
    all_years.append(year)

# Creating an empty Pandas DataFrame covering the whole range of data 
year_start = min(all_years) 
end_year = max(all_years)
date_range = pd.date_range(start = str(year_start) + '-01-01', 
                           end = str(end_year) + '-12-31', 
                           freq = 'D')
df = pd.DataFrame(0.0, columns = ['Temparature'], index = date_range)
    
# Defining the location, lat, lon based on the csv data
cities = pd.read_csv('Cities.csv')

for index, row in cities.iterrows():
    location = row['Name']
    location_latitude = row['Latitude']
    location_longitude = row['Longitude']

    # Sorting the all_years python list
    all_years.sort()
    
    for yr in all_years:
        # Reading-in the data 
        data = Dataset(str(yr)+'.nc', 'r')
        
        # Storing the lat and lon data of the netCDF file into variables 
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]
        
        # Squared difference between the specified lat,lon and the lat,lon of the netCDF 
        sq_diff_lat = (lat - location_latitude)**2 
        sq_diff_lon = (lon - location_longitude)**2
        
        # Identify the index of the min value for lat and lon
        min_index_lat = sq_diff_lat.argmin()
        min_index_lon = sq_diff_lon.argmin()
        
        # Accessing the average temparature data
        temp = data.variables['tave']
        
        # Creating the date range for each year during each iteration
        start = str(yr) + '-01-01'
        end = str(yr) + '-12-31'
        d_range = pd.date_range(start = start, 
                                end = end, 
                                freq = 'D')
        
        for t_index in np.arange(0, len(d_range)):
            print('Recording the value for '+ location+': ' + str(d_range[t_index]))
            df.loc[d_range[t_index]]['Temparature'] = temp[t_index, min_index_lat, min_index_lon]
    
    df.to_csv(location +'.csv')
    
    
    
    
    
    
    
    