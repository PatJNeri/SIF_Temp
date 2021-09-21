# File to run thru the geo-location of datasets and produce a clear
# description of the data location
# 9-19-21
# Patrick Neri
# %%
# Currently running on hastools2
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
# %%
PSIIGEO = PSIIContr[PSIIContr['GEO'].notna() == True]
# %%
# Here we need to formalize and split 'GEO' values to lat and lon
PSIIGEO['lat_split'] = PSIIGEO['GEO']
PSIIGEO['lon_split'] = PSIIGEO['GEO']
potent_error = []
for i in range(0, len(PSIIGEO['GEO'])):
    print(i)
    if len(str(PSIIGEO['GEO'].iloc[i]).split(',')) == 2:
        PSIIGEO['lat_split'].iloc[i] = str(PSIIGEO['GEO'].iloc[i]).split(',')[0]
        PSIIGEO['lon_split'].iloc[i] = str(PSIIGEO['GEO'].iloc[i]).split(',')[1]
    else:
        potent_error.append(i)

PSIIGEO['lat_num'] = PSIIGEO['lat_split']
for i in range(0, len(PSIIGEO['GEO'])):
    print(i)
    setnums = re.split('[°\'"]', PSIIGEO['lat_split'].iloc[i])
    print(setnums)
    if len(setnums) == 2:
        PSIIGEO['lat_num'].iloc[i] == (float(setnums[0]) * (-1 if setnums[1] in ['W', 'S'] else 1))
    elif len(setnums) == 3:
        PSIIGEO['lat_num'].iloc[i] = (float(setnums[0]) + float(setnums[1])/60) * (-1 if setnums[2] in ['W', 'S'] else 1)
    elif len(setnums) == 4:
        PSIIGEO['lat_num'].iloc[i] = (float(setnums[0]) + float(setnums[1])/60 + float(setnums[2])/(60*60)) * (-1 if setnums[3] in ['W', 'S'] else 1)
# %%
# Here we need to plot them
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()


plt.show()
# %%
deg, minutes, seconds, direction