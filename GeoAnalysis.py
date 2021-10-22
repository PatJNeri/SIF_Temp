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
from pandas.core.algorithms import rank
import scipy.stats as stt
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
# Optional method to output for desktop confirmation
#f.to_csv('c:/Users/pjneri/Desktop/f.csv', na_rep='Unknown')
# %%
# Used the HPC_climatology.py basis to get out values from data set provided by Prof Song
# These are the files:
Precip = pd.read_csv('Hist_Precip.csv')
Qbot = pd.read_csv('Hist_Qbot.csv')
Solar = pd.read_csv('Hist_Solar.csv')
TempB = pd.read_csv('Hist_TempB.csv')

# Convert to DataFrame
P = pd.DataFrame(data=Precip)
Q = pd.DataFrame(data=Qbot)
S = pd.DataFrame(data=Solar)
T = pd.DataFrame(data=TempB)
# %%
# Need to form a way to convert form the string date column to a more
# usabe datetime method for ease of call and averaging.
P['time'] = pd.to_datetime(P['time'])
Q['time'] = pd.to_datetime(Q['time'])
S['time'] = pd.to_datetime(S['time'])
T['time'] = pd.to_datetime(T['time'])
# %%
# Making year accumulation set for precipitation
# https://stackoverflow.com/questions/29370057/select-dataframe-rows-between-two-dates
# https://stackoverflow.com/questions/39158699/how-to-select-column-and-rows-in-pandas-without-column-or-row-names
cum_ann_Precip = np.zeros((32, 35))
for l in range(1985, 2017):
    strt = str(l) + '0101'
    endd = str(l+1) + '0101'
    mask = (P['time'] > pd.to_datetime(strt, format='%Y%m%d')) & (P['time'] < pd.to_datetime(endd, format='%Y%m%d'))
    year_ = P.loc[mask]
    for j in range(0,35):
        cum_ann_Precip[l-1985, j] = sum(year_.iloc[:, j+2]*21600)

# This could be refined, currently just lists month averages in chrono order
# to call specific months...
monthly_Precip = np.zeros((384, 35))
for l in range(1985, 2017):
    for m in range(1,13):
        if m < 9:
            strt = str(l) + '0' + str(m) + '01'
            endd = str(l) + '0' + str(m+1) + '01'
        elif m == 9:
            strt = str(l) + '0901'
            endd = str(l) + '1001'
        elif m < 12:
            strt = str(l) + str(m) + '01'
            endd = str(l) + str(m+1) + '01'
        else:
            strt = str(l) + '1201'
            endd = str(l+1) + '0101'
        mask = (P['time'] > pd.to_datetime(strt, format='%Y%m%d')) & (P['time'] < pd.to_datetime(endd, format='%Y%m%d'))
        month_ = P.loc[mask]
        for j in range(0,35):
            monthly_Precip[(l-1985)*12 + m - 1, j] = sum(month_.iloc[:, j+2])
# %%
# trying a different method
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/09_timeseries.html?highlight=datetime
# https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html
day_prec_try = P.groupby(P['time'].dt.date).mean()
month_temp_try = T.groupby(T['time'].dt.month).mean()
# %%
# Produce a method that orders the monthly values of whichever
# location, then do a [:3] for top and [(len(timeseries)-3):]
ranked_3mon_sum = np.zeros(12)
print('1') 
ranked_3mon_sum[0] = (np.mean((month_temp_try['location 0'][1], month_temp_try['location 0'][2], month_temp_try['location 0'][12])))
for i in range(2,12):
    print(i)
    ranked_3mon_sum[i-1] = (np.mean(month_temp_try['location 0'][i-1:i+1]))
print('12')
ranked_3mon_sum[11] = (np.mean((month_temp_try['location 0'][11], month_temp_try['location 0'][12], month_temp_try['location 0'][1])))
# Finding max 3-month running average to define 'season'
ranked_df = pd.DataFrame(data=ranked_3mon_sum)
ranked_df['month'] = ranked_df.index + 1
ranked_df['rank'] = ranked_df.iloc[:,0].rank()

peak_month = ranked_df[ranked_df['rank'] == 12]['month']
weak_month = ranked_df[ranked_df['rank'] == 1]['month']
# Creating histogram groups
if int(peak_month) == 1:
    peak_hist_set = T[T['time'].dt.month.isin([12,1,2])]['location 0']
elif int(peak_month) == 12:
    peak_hist_set = T[T['time'].dt.month.isin([11,12,1])]['location 0']
else:
    peak_hist_set = T[T['time'].dt.month.isin([int(peak_month-1), int(peak_month), int(peak_month+1)])]['location 0']
if int(weak_month) == 1:
    weak_hist_set = T[T['time'].dt.month.isin([12,1,2])]['location 0']
elif int(weak_month) == 12:
    weak_hist_set = T[T['time'].dt.month.isin([11,12,1])]['location 0']
else:
    weak_hist_set = T[T['time'].dt.month.isin([int(weak_month-1), int(weak_month), int(weak_month+1)])]['location 0']
# Getting the 'historic highs and lows' and means
his_dis_weak = stt.rv_histogram(np.histogram(weak_hist_set, bins=20))
his_dis_peak = stt.rv_histogram(np.histogram(peak_hist_set, bins=20))

weak_mean = his_dis_weak.ppf(0.5)
weak_lower = his_dis_weak.ppf(0.25)
peak_mean = his_dis_peak.ppf(0.5)
peak_upper = his_dis_peak.ppf(0.75)
clim_mean = np.mean(T['location 0'])
# %%
# methods of saving them (building a loop)
clim_locs_historic = pd.DataFrame()
for j in range(0,35):
    location = 'location ' + str(j)
    print(location)
    ranked_3mon_sum = np.zeros(12)
    ranked_3mon_sum[0] = (np.mean((month_temp_try[location][1], month_temp_try[location][2], month_temp_try[location][12])))
    for i in range(2,12):
        ranked_3mon_sum[i-1] = (np.mean(month_temp_try[location][i-1:i+1]))
    ranked_3mon_sum[11] = (np.mean((month_temp_try[location][11], month_temp_try[location][12], month_temp_try[location][1])))
    # Finding max 3-month running average to define 'season'
    ranked_df = pd.DataFrame(data=ranked_3mon_sum)
    ranked_df['month'] = ranked_df.index + 1
    ranked_df['rank'] = ranked_df.iloc[:,0].rank()
    ranked_df
    peak_month = ranked_df[ranked_df['rank'] == 12]['month']
    weak_month = ranked_df[ranked_df['rank'] == 1]['month']
    # Creating histogram groups
    if int(peak_month) == 1:
        peak_hist_set = T[T['time'].dt.month.isin([12,1,2])][location]
    elif int(peak_month) == 12:
        peak_hist_set = T[T['time'].dt.month.isin([11,12,1])][location]
    else:
        peak_hist_set = T[T['time'].dt.month.isin([int(peak_month-1), int(peak_month), int(peak_month+1)])][location]
    if int(weak_month) == 1:
        weak_hist_set = T[T['time'].dt.month.isin([12,1,2])][location]
    elif int(weak_month) == 12:
        weak_hist_set = T[T['time'].dt.month.isin([11,12,1])][location]
    else:
        weak_hist_set = T[T['time'].dt.month.isin([int(weak_month-1), int(weak_month), int(weak_month+1)])][location]
    # Getting the 'historic highs and lows' and means
    his_dis_weak = stt.rv_histogram(np.histogram(weak_hist_set, bins=20))
    his_dis_peak = stt.rv_histogram(np.histogram(peak_hist_set, bins=20))
    # Components to add to dataframe
    weak_mean = his_dis_weak.ppf(0.5)
    weak_lower = his_dis_weak.ppf(0.25)
    peak_mean = his_dis_peak.ppf(0.5)
    peak_upper = his_dis_peak.ppf(0.75)
    clim_mean = np.mean(T[location])
    clim_locs_historic[j] = [int(weak_month), weak_mean, weak_lower, int(peak_month), peak_mean, peak_upper, clim_mean]
# %%
# Plot method to show the variation over the years
loc_num = 7
r = np.arange(0, 1, (1/366))
r1 = np.arange(0,1, (1/12))
theta = 2 * np.pi * r
theta1 = 2 * np.pi * r1
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, day_prec_try.iloc[:, loc_num])
#ax.plot(theta1, month_temp_try.iloc[:,loc_num])
#ax.plot(theta1, )
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()

# %%
plt.hist(T['location 2'][:])
# %%
# For precip, determine the mean and variance for each location 
# to serve as a numerical proxy of climatological history

# Display of the values of each locations annual accum. precip.
for i in range(0,35):
    plt.plot(range(0,32), cum_ann_Precip[:, i])

# Make a list of the means and std for each location
precip_clim = np.zeros((35,2))
for i in range(0,35):
    precip_clim[i, 1] = np.std(cum_ann_Precip[:, i])
    precip_clim[i, 0] = np.mean(cum_ann_Precip[:, i])
# %%
for i in range(0,35):
    plt.plot(range(0,32), cum_ann_Tbot[:, i])

# Make a list of the means and std for each location
Tbot_clim = np.zeros((35,2))
for i in range(0,35):
    Tbot_clim[i, 1] = np.std(cum_ann_Tbot[:, i])
    Tbot_clim[i, 0] = np.mean(cum_ann_Tbot[:, i])
# %%
 