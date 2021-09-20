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
for i in range(len(PSIIGEO['GEO'])):
    PSIIGEO['lat_split'].iloc[i] = str(PSIIGEO['GEO'].iloc[i]).split(',')[0]
    PSIIGEO['lon_split'].iloc[i] = str(PSIIGEO['GEO'].iloc[i]).split(',')[1]
# %%
# Here we need to plot them
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()


plt.show()