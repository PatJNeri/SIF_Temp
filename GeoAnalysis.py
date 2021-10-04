# File to run thru the geo-location of datasets and produce a clear
# description of the data location
# 9-19-21
# Patrick Neri
# %%
# Currently running on hastools2, Outputs Locs File 'f.csv'
from matplotlib import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import re
# %%
# Import dataset and fix some formatting
print(os.getcwd())
PSIImaster = pd.read_excel('PSIImax-Master2-24.xlsx', engine='openpyxl')
PSIImaster['Clade A'] = PSIImaster['Clade A'].astype(str)
PSIImaster['Clade B'] = PSIImaster['Clade B'].astype(str)
PSIImaster['Order'] = PSIImaster['Order'].astype(str)
PSIImaster['Family'] = PSIImaster['Family'].astype(str)
# %%
# Divide up into subplot datasets
PSIImaster['HeatMid'] = (PSIImaster['HeatUp'] + PSIImaster['HeatDown'])/2
PSIImaster['Heatrange'] = PSIImaster['HeatUp'] - PSIImaster['HeatDown']
PSIImaster.name = 'PSII Master'

PSIIContr = PSIImaster[(PSIImaster['water status'] == 0) & (PSIImaster['nut status'] == 0)]
PSIIContr.name = 'PSII master Control'
PSIICCrop = PSIIContr[PSIIContr['type'] == 'crop']
PSIICCrop.name = 'PSII control Crop'
PSIICTree = PSIIContr[PSIIContr['type'] == 'tree']
PSIICTree.name = 'PSII control Tree'
PSIICGrass = PSIIContr[PSIIContr['type'] == 'grass-like']
PSIICGrass.name = 'PSII control Grass-like'
PSIICShrub = PSIIContr[PSIIContr['type'] == 'shrub']
PSIICShrub.name = 'PSII control shrub'

PSIIContr['Adjusted PFT'] = PSIIContr['PFT #']
for i in range(len(PSIIContr['phiPSIImax'])):
    if PSIIContr['PFT #'].iloc[i] == 71:
        PSIIContr['Adjusted PFT'].iloc[i] = 14
    if PSIIContr['PFT #'].iloc[i] == 73:
        PSIIContr['Adjusted PFT'].iloc[i] = 14
    if PSIIContr['PFT #'].iloc[i] == 19:
        PSIIContr['Adjusted PFT'].iloc[i] = 15
    if PSIIContr['PFT #'].iloc[i] == 23:
        PSIIContr['Adjusted PFT'].iloc[i] = 15
    if PSIIContr['PFT #'].iloc[i] == 61:
        PSIIContr['Adjusted PFT'].iloc[i] = 15
    if PSIIContr['PFT #'].iloc[i] == 41:
        PSIIContr['Adjusted PFT'].iloc[i] = 15
    if PSIIContr['PFT #'].iloc[i] == 17:
        PSIIContr['Adjusted PFT'].iloc[i] = 16
    if PSIIContr['PFT #'].iloc[i] == 51:
        PSIIContr['Adjusted PFT'].iloc[i] = 16
    if PSIIContr['PFT #'].iloc[i] == 63:
        PSIIContr['Adjusted PFT'].iloc[i] = 16
    if PSIIContr['PFT #'].iloc[i] == 39:
        PSIIContr['Adjusted PFT'].iloc[i] = 4
    if PSIIContr['PFT #'].iloc[i] == 53:
        PSIIContr['Adjusted PFT'].iloc[i] = 4

PSIIContr['timetime'] = PSIIContr['Time (h)']
for i in range(len(PSIIContr['Time (h)'])):
    if PSIIContr['Time (h)'].iloc[i] == 0:
        PSIIContr['timetime'].iloc[i] = 0
    elif ((PSIIContr['Time (h)'].iloc[i] > 0) & (PSIIContr['Time (h)'].iloc[i] < 48.1)):
        PSIIContr['timetime'].iloc[i] = 1
    elif ((PSIIContr['Time (h)'].iloc[i] > 48) & (PSIIContr['Time (h)'].iloc[i] < 336.1)):
        PSIIContr['timetime'].iloc[i] = 2
    elif PSIIContr['Time (h)'].iloc[i] > 336:
        PSIIContr['timetime'].iloc[i] = 3
# %%
# drop bad data options
NewSet = PSIIContr.drop([250,251,252,253,1502,1503,1504,1505,1506,1507,1508,1509,1510,1511], axis=0)
PSIIGEO = NewSet[NewSet['GEO'].notna() == True]
PSIIGEO['index'] = PSIIGEO.index
print(np.unique(PSIIGEO['paper']))
# %%
# Here we need to formalize and split 'GEO' values to lat and lon
PSIIGEO['lat_split'] = PSIIGEO['GEO']
PSIIGEO['lon_split'] = PSIIGEO['GEO']
potent_error = []
for i in range(0, len(PSIIGEO['GEO'])):
    if len(str(PSIIGEO['GEO'].iloc[i]).split(',')) == 2:
        PSIIGEO['lat_split'].iloc[i] = str(PSIIGEO['GEO'].iloc[i]).split(',')[0]
        PSIIGEO['lon_split'].iloc[i] = str(PSIIGEO['GEO'].iloc[i]).split(',')[1]
    else:
        potent_error.append(i)

PSIIGEO['lat_num'] = PSIIGEO['lat_split']
for i in range(0, len(PSIIGEO['GEO'])):
    setnums = re.split('[°\'"]', PSIIGEO['lat_split'].iloc[i])
    if len(setnums) == 2:
        PSIIGEO['lat_num'].iloc[i] = (float(setnums[0])) * (-1 if setnums[1] in ['W', 'S', ' W', ' S'] else 1)
    elif len(setnums) == 3:
        PSIIGEO['lat_num'].iloc[i] = (float(setnums[0]) + float(setnums[1])/60) * (-1 if setnums[2] in ['W', 'S', ' W', ' S'] else 1)
    elif len(setnums) == 4:
        PSIIGEO['lat_num'].iloc[i] = (float(setnums[0]) + float(setnums[1])/60 + float(setnums[2])/(60*60)) * (-1 if setnums[3] in ['W', 'S', ' W', ' S'] else 1)

PSIIGEO['lon_num'] = PSIIGEO['lon_split']
for i in range(0, len(PSIIGEO['GEO'])):
    setnums = re.split('[°\'"]', PSIIGEO['lon_split'].iloc[i])
    if len(setnums) == 2:
        PSIIGEO['lon_num'].iloc[i] = (float(setnums[0])) * (-1 if setnums[1] in ['W', 'S', ' W', ' S'] else 1)
    elif len(setnums) == 3:
        PSIIGEO['lon_num'].iloc[i] = (float(setnums[0]) + float(setnums[1])/60) * (-1 if setnums[2] in ['W', 'S', ' W', ' S'] else 1)
    elif len(setnums) == 4:
        PSIIGEO['lon_num'].iloc[i] = (float(setnums[0]) + float(setnums[1])/60 + float(setnums[2])/(60*60)) * (-1 if setnums[3] in ['W', 'S', ' W', ' S'] else 1)        
# %%
# Here we need to plot them
# Finally fixed!!!

ax = plt.axes(projection=ccrs.PlateCarree())
for i in range(0, len(PSIIGEO['GEO'])):
    ax.plot(PSIIGEO['lon_num'].iloc[i], PSIIGEO['lat_num'].iloc[i], 'bo', markersize=2)
    #ax.text(float(PSIIGEO['lon_num'].iloc[i]) + 0.01, float(PSIIGEO['lat_num'].iloc[i]) - 0.01, PSIIGEO['paper'].iloc[i], transform=ccrs.PlateCarree())

#ax.coastlines(color='black')
ax.stock_img()
plt.show()
# %%
# generating a uniform string that can be more easily checked against other locations
PSIIGEO['latlon'] = PSIIGEO['lon_num']
for i in range(0, len(PSIIGEO['lat_num'])):
    PSIIGEO['latlon'].iloc[i] = (str(PSIIGEO['lat_num'].iloc[i]) + ', ' + str(PSIIGEO['lon_num'].iloc[i]))

# IMPORTANT!!
# The method used below may not work if 2 locations are listed identical
# but come from different sources. Need a more wholistic method to 
# avoid this issue.

# %%
# Make lists that can be checked (potentially not necessary)
locations = []
isss = []
for i in range(0, len(PSIIGEO['latlon'])):
    if PSIIGEO['latlon'].iloc[i] in locations:
        continue
    else:
        locations.append(PSIIGEO['latlon'].iloc[i])
        isss.append(i)

# %%
# Ensure that all the locations are captured, and by this, all the necessary Data
print(len(locations) == len(np.unique(PSIIGEO['latlon'])))
for locs in locations:
    if locs in np.unique(PSIIGEO['latlon']):
        continue
    else:
        print(locs, 'Not represented') 

# %%
# Now make a new dataframe with just the rows of unique locations
# https://stackoverflow.com/questions/34682828/extracting-specific-selected-columns-to-new-dataframe-as-a-copy
# Resetting the index here to make appropiate call
PSIIGEO = PSIIGEO.reset_index(drop=True)
# Both these methods work, will use the first one for now
f = PSIIGEO.filter(isss, axis=0)
#f1 = PSIIGEO[PSIIGEO.index.isin(isss)]
# Generate a csv out of this new dataframe
# https://towardsdatascience.com/how-to-export-pandas-dataframe-to-csv-2038e43d9c03
f.to_csv('c:/Users/PJN89/Desktop/f.csv', na_rep='Unkown')
# %%
