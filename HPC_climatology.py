# HPC file

import pandas as pd
import xarray as xr

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
        name = 'location ' + str(i)
        lat_ind = int((improve_coord(f['lat_num'].iloc[i]) + 89.75)/0.5)
        lon_ind = int((improve_coord(f['lon_num'].iloc[i]) + 179.75)/0.5)
        Trialfile[name] = ncdir.variables['PRECTmms'][:, lat_ind, lon_ind]
    return Trialfile


locations = pd.read_csv('c:/Users/pjneri/Desktop/f.csv')
Hist_data = pd.DataFrame()
for k in range(2016, 2017):
    for i in range(10, 13):
        dfname = "df-" + str(k) + '-' + str(i)
        print(dfname)
        df = xr.load_dataset('c:/Users/pjneri/Desktop/clmforc.cruncep.V7.c2016.0.5d.Prec.2016-' + str(i) + '.nc')
        df_locs = location_values(df, locations)
        print(df_locs.shape)
        Hist_data = Hist_data.append(df_locs, ignore_index=True)

Hist_data.to_csv('c:/Users/pjneri/Desktop/Hist_Precip.csv', na_rep='Unknown')
# %%
