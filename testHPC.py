# Test for HPC code
# Neri 9-26-21

# %%
# from GeoAnalysis import PSIIGEO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from netCDF4 import Dataset
# https://github.com/nco/pynco
# https://joehamman.com/2014/01/29/Python-Bindings-for-NCO/
# https://linux.die.net/man/1/ncks
# http://nco.sourceforge.net/nco.html#ncks
#from nco import Nco

# %%
def coord_round(x):
    return ((round((x-0.25)*4)/4) + 0.25)

def improve_coord(x):
    if x > round(x):
        return (round(x) + 0.25)
    else:
        return (round(x) - 0.25)

def location_values(ncdir, f):
    """This is a great function, who needs nco"""
    Trialfile = pd.DataFrame()
    num_of_loc = len(f.index)
    Trialfile['time'] = ncdir.coords["time"][:]
    for i in range(0, num_of_loc):
        print(i)
        name = 'location ' + str(i)
        lat_ind = int((improve_coord(f['lat_num'].iloc[i]) + 89.75)/0.5)
        lon_ind = int((improve_coord(f['lon_num'].iloc[i]) + 179.75)/0.5)
        Trialfile[name] = ncdir.variables['PRECTmms'][:, lat_ind, lon_ind]
    return Trialfile
# %%
# direct import first trial no loop files
#nco = Nco()
print(os.getcwd())

data10 = xr.load_dataset('c:/Users/pjneri/Desktop/clmforc.cruncep.V7.c2016.0.5d.Prec.2016-10.nc')
locations = pd.read_csv('c:/Users/pjneri/Desktop/f.csv')

# %%
# Practice for pulling from a different directory
path = "c:/Users/PJN89/temp_git"

# Make sure the file path exists
os.path.exists("c:/Users/PJN89/temp_git/PSIImax-Master2-24.xlsx")

# Eventual goal cd file
PSIImaster = pd.read_excel(os.path.join(path, 'PSIImax-Master2-24.xlsx'), engine='openpyxl')

# %%
# Run thru for Precip6hourly
print(os.getcwd())
os.path.exists("c:/Users/PJN89/Desktop/clmforc.cruncep.V7.c2016.0.5d.Prec.2016-12.nc")
# %%
Hist_data = pd.DataFrame()
for i in range(10, 13):
    dfname = "df" + str(i)
    print(dfname)
    df = xr.load_dataset('c:/Users/pjneri/Desktop/clmforc.cruncep.V7.c2016.0.5d.Prec.2016-' + str(i) + '.nc')
    df_locs = location_values(df, locations)
    print(df_locs.shape)
    Hist_data = Hist_data.append(df_locs, ignore_index=True)
# %%
# https://gis.stackexchange.com/questions/327921/extracting-lat-long-numeric-value-of-one-pixel-for-all-variables-in-netcdf4

# %%
# https://scitools.org.uk/cartopy/docs/v0.11/matplotlib/advanced_plotting.html

sst = data10.variables['PRECTmms'][0, :, :]
lats = data10.variables['lat'][:]
lons = data10.variables['lon'][:]

ax = plt.axes(projection=ccrs.PlateCarree())

plt.contourf(lons, lats, sst, 60,
             transform=ccrs.PlateCarree())

ax.coastlines()

plt.show()
# %%
for i in range(1,17):
    plt.plot(PSIIGEO['HeatMid'][PSIIGEO['Adjusted PFT'] == i],
    PSIIGEO['phiPSIImax'][PSIIGEO['Adjusted PFT'] == i], 'o')
    plt.title(i)
    plt.show()
# %%

# %%
