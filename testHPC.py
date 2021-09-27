# Test for HPC code
# Neri 9-26-21

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from netCDF4 import Dataset

# %%
print(os.getcwd())

#data10 = xr.load_dataset('clmforc.cruncep.V7.c2016.0.5d.Prec.2016-10.nc')

# %%
# Practice for pulling from a different directory
path = "c:/Users/PJN89/temp_git"

# Make sure the file path exists
os.path.exists("c:/Users/PJN89/temp_git/PSIImax-Master2-24.xlsx")

# Eventual goal cd file
PSIImaster = pd.read_excel(os.path.join(path, 'PSIImax-Master2-24.xlsx'), engine='openpyxl')

# %%
# Run thru for Precip6hourly


# %%
for i in range(10, 13):
    name = "ds" + str(i)
    print(name)
    name = xr.load_dataset('clmforc.cruncep.V7.c2016.0.5d.Prec.2016-' + str(i) + '.nc')
# %%
# https://gis.stackexchange.com/questions/327921/extracting-lat-long-numeric-value-of-one-pixel-for-all-variables-in-netcdf4

def ExtractVarsFromNetcdf(x_coord, y_coord, ncdir, varnames):
    """   
    @params:
        x_coord    - Required : the x coordinate of the point
        x_coord    - Required : the y coordinate of the point
        ncdir      - Required : The directory of the netcdf file.
        varnames   - Required : The netcdf variables
    """

    with Dataset(ncdir, "r") as nc:

        # Get the nc lat and lon from the point's x, y
        lon = nc.variables["lon"][int(round(x_coord))]
        lat = nc.variables["lat"][int(round(y_coord))]

        # Return a np.array with the netcdf data
        nc_data = np.ma.getdata(
            [nc.variables[varname][:, x_coord, y_coord] for varname in varnames]
        )

        return nc_data, lon, lat
        

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
