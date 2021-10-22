# ClimatologyStats
# Patrick Neri
# LETS DO THIS
# hastools2
# %%
from numpy.core.function_base import linspace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import math
from matplotlib import transforms
import scipy.stats as stt
import cartopy.crs as ccrs
import re
# %%
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

# only including PFT for checking things later
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

PSIIContr['temptemp'] = PSIIContr['HeatMid']
for i in range(len(PSIIContr['HeatMid'])):
        if (PSIIContr['HeatMid'].iloc[i] < 0):
                PSIIContr['temptemp'].iloc[i] = 0
        elif ((PSIIContr['HeatMid'].iloc[i] > 0) & (PSIIContr['HeatMid'].iloc[i] < 10.1)):
                PSIIContr['temptemp'].iloc[i] = 1
        elif ((PSIIContr['HeatMid'].iloc[i] > 10) & (PSIIContr['HeatMid'].iloc[i] < 35.1)):
                PSIIContr['temptemp'].iloc[i] = 2
        elif ((PSIIContr['HeatMid'].iloc[i] > 35) & (PSIIContr['HeatMid'].iloc[i] < 45.1)):
                PSIIContr['temptemp'].iloc[i] = 3
        elif (PSIIContr['HeatMid'].iloc[i] > 45):
                PSIIContr['temptemp'].iloc[i] = 4

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
# generating a uniform string that can be more easily checked against other locations
PSIIGEO['latlon'] = PSIIGEO['lon_num']
for i in range(0, len(PSIIGEO['lat_num'])):
    PSIIGEO['latlon'].iloc[i] = (str(PSIIGEO['lat_num'].iloc[i]) + ', ' + str(PSIIGEO['lon_num'].iloc[i]))

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
# Gather and form values to covert into new columns in main PSII dataframe
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

month_temp_try = T.groupby(T['time'].dt.month).mean()

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

#Friedman
low_mon_num = np.histogram_bin_edges(clim_locs_historic[0,:], bins='df')
low_mon_mean = np.histogram_bin_edges(clim_locs_historic[1,:], bins='df')
low_mon_low = np.histogram_bin_edges(clim_locs_historic[2,:], bins='df')
high_mon_num = np.histogram_bin_edges(clim_locs_historic[3,:], bins='df')
high_mon_mean = np.histogram_bin_edges(clim_locs_historic[4,:], bins='df')
high_mon_low = np.histogram_bin_edges(clim_locs_historic[5,:], bins='df')
clim_allmean = np.histogram_bin_edges(clim_locs_historic[6,:], bins='df')








# %%
# %%
# Below is the raw method to perform 3 way ANOVA
PSII3ANOVA = PSIIGEO
PSII3ANOVA.dropna(subset=['Adjusted PFT'])
PSII3ANOVA.dropna(subset=['temptemp'], inplace=True)
PSII3ANOVA.dropna(subset=['timetime'], inplace=True)
# Let Temp be i, time be j, PFT be k
Txtime = np.zeros([5, 4])
for i in range(0,5):
    for j in range(0,4):
        Txtime[i,j] = np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['temptemp'] == i) & (PSII3ANOVA['timetime'] == j)])

TxSpec = np.zeros([5,16])
for i in range(0,5):
    for k in range(1,17):
        TxSpec[i,k-1] = np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['temptemp'] == i) & (PSII3ANOVA['Adjusted PFT'] == k)])

timexSpec = np.zeros([4,16])
for j in range(0,4):
    for k in range(1,17):
        timexSpec[j,k-1] = np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['timetime'] == j) & (PSII3ANOVA['Adjusted PFT'] == k)])

Tsolo = np.zeros(5)
for i in range(0, 5):
    Tsolo[i] = np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['temptemp'] == i)])

Timesolo = np.zeros(4)
for j in range(0,4):
    Timesolo[j] = np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['timetime'] == j)])

Speciessolo = np.zeros(16)
for i in range(1,17):
    Speciessolo[k-1] = np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['Adjusted PFT'] == k)])

mu = np.mean(PSII3ANOVA['phiPSIImax'])
#PSII3ANOVA.dropna(axis=0, subset=['timetime'])

PSII3ANOVA['ART T way'] = PSII3ANOVA['phiPSIImax']
PSII3ANOVA['ART t way'] = PSII3ANOVA['phiPSIImax']
PSII3ANOVA['ART S way'] = PSII3ANOVA['phiPSIImax']
PSII3ANOVA['ART Tt way'] = PSII3ANOVA['phiPSIImax']
PSII3ANOVA['ART TS way'] = PSII3ANOVA['phiPSIImax']
PSII3ANOVA['ART tS way'] = PSII3ANOVA['phiPSIImax']
PSII3ANOVA['ART TtS way'] = PSII3ANOVA['phiPSIImax']
# The following line drops elements of PFT 13 based on the input of timetime, make sure to remake the 
# dataframe if this is not a relevant commponent of ongoing test
PSII3ANOVA.dropna(subset=['timetime'], inplace=True)

for x in range(len(PSII3ANOVA['phiPSIImax'])):
    #print(x)
    i = int(PSII3ANOVA['temptemp'].iloc[x])
    #print(i)
    j = int(PSII3ANOVA['timetime'].iloc[x])
    #print(j)
    k = int(PSII3ANOVA['Adjusted PFT'].iloc[x])
    #print(k)
    PSII3ANOVA['ART T way'].iloc[x] = (PSII3ANOVA['phiPSIImax'].iloc[x] - np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['temptemp'] == i) & (PSII3ANOVA['timetime'] == j) & (PSII3ANOVA['Adjusted PFT'] == k)]) + Tsolo[i] - mu)
    PSII3ANOVA['ART t way'].iloc[x] = (PSII3ANOVA['phiPSIImax'].iloc[x] - np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['temptemp'] == i) & (PSII3ANOVA['timetime'] == j) & (PSII3ANOVA['Adjusted PFT'] == k)]) + Timesolo[j] - mu)
    PSII3ANOVA['ART S way'].iloc[x] = (PSII3ANOVA['phiPSIImax'].iloc[x] - np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['temptemp'] == i) & (PSII3ANOVA['timetime'] == j) & (PSII3ANOVA['Adjusted PFT'] == k)]) + Speciessolo[k-1] - mu)
    PSII3ANOVA['ART Tt way'].iloc[x] = (PSII3ANOVA['phiPSIImax'].iloc[x] - np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['temptemp'] == i) & (PSII3ANOVA['timetime'] == j) & (PSII3ANOVA['Adjusted PFT'] == k)]) + Txtime[i,j] - Tsolo[i] - Timesolo[j] + mu)
    PSII3ANOVA['ART TS way'].iloc[x] = (PSII3ANOVA['phiPSIImax'].iloc[x] - np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['temptemp'] == i) & (PSII3ANOVA['timetime'] == j) & (PSII3ANOVA['Adjusted PFT'] == k)]) + TxSpec[i,k-1] - Tsolo[i] - Speciessolo[k-1] + mu)
    PSII3ANOVA['ART tS way'].iloc[x] = (PSII3ANOVA['phiPSIImax'].iloc[x] - np.mean(PSII3ANOVA['phiPSIImax'][(PSII3ANOVA['temptemp'] == i) & (PSII3ANOVA['timetime'] == j) & (PSII3ANOVA['Adjusted PFT'] == k)]) + timexSpec[j,k-1] - Timesolo[j] - Speciessolo[k-1] + mu)
    PSII3ANOVA['ART TtS way'].iloc[x] = (PSII3ANOVA['phiPSIImax'].iloc[x] - Txtime[i,j] - TxSpec[i,k-1] - timexSpec[j,k-1] + Tsolo[i] + Timesolo[j] + Speciessolo[k-1] - mu)


PSIITranked3Way = PSII3ANOVA.sort_values(by='ART T way')
PSIITranked3Way['number'] = PSIITranked3Way['ART T way'].rank()
PSIItranked3Way = PSII3ANOVA.sort_values(by='ART t way')
PSIItranked3Way['number'] = PSIItranked3Way['ART t way'].rank()
PSIISranked3Way = PSII3ANOVA.sort_values(by='ART S way')
PSIISranked3Way['number'] = PSIISranked3Way['ART S way'].rank()
PSIITtranked3Way = PSII3ANOVA.sort_values(by='ART Tt way')
PSIITtranked3Way['number'] = PSIITtranked3Way['ART Tt way'].rank()
PSIITSranked3Way = PSII3ANOVA.sort_values(by='ART TS way') 
PSIITSranked3Way['number'] = PSIITSranked3Way['ART TS way'].rank()
PSIItSranked3Way = PSII3ANOVA.sort_values(by='ART tS way')
PSIItSranked3Way['number'] = PSIItSranked3Way['ART tS way'].rank()
PSIIranked3Way = PSII3ANOVA.sort_values(by='ART TtS way')
PSIIranked3Way['number'] = PSIIranked3Way['ART TtS way'].rank()

# %%
Pogg = PSIIranked3Way
dCont = {'rank': Pogg['number'], 'Temp': Pogg['temptemp'], 'Time': Pogg['timetime'], 'Species': Pogg['Adjusted PFT']}
DfCont = pd.DataFrame(data=dCont)
model = ols('rank ~ C(Temp) + C(Time) + C(Species) + C(Temp):C(Time) + C(Temp):C(Species) + C(Time):C(Species) + C(Temp):C(Time):C(Species) -1', data=DfCont).fit()
anova_table = sm.stats.anova_lm(model, typ=3)
anova_table
# %%
