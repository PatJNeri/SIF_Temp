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
from lmfit.models import RectangleModel
import re
import random
# %%
def get_Mod_paramsValues(output):
        """Getting out values of model params with numerical values
        Produces a dataframe of param names and values """
        Var_list = output.var_names
        frame = []
        for n in range(len(Var_list)):
                vari = output.params[Var_list[n]]
                vari_split = str(vari).split()
                vari_value = vari_split[2].split('=')
                frame.append((Var_list[n], float(vari_value[1][:-1])))
        Paramet = pd.DataFrame(frame, columns=['name', 'val'])
        return Paramet

def get_set_resid(dataset):
        """ Pick a chosen dataset that has a HeatMid column and phiPSIImax column"""
        Ordered = dataset.sort_values(by='HeatMid')
        x = Ordered['HeatMid']
        x0 = x.iloc[:]
        y = Ordered['phiPSIImax']
        y0 = y.iloc[:]
        rang = [np.min(x), np.max(x)]
        # Quad Model run
        mod = RectangleModel(form='erf')
        mod.set_param_hint('amplitude', value=0.8, min=0.75, max=0.9)
        mod.set_param_hint('center1', value=-3, min=-15, max=10)
        mod.set_param_hint('center2', value=45, min=15, max=60)
        mod.set_param_hint('sigma1', value=7, min=1, max=12)
        mod.set_param_hint('sigma2', value=7, min=1, max=12)
        pars = mod.guess(y0, x=x0)
        pars['amplitude'].set(value=0.8, min=0.6, max=0.83)
        pars['center1'].set(value=-6, min=-23, max=7)
        pars['center2'].set(value=46, min=35, max=57)
        pars['sigma1'].set(value=7, min=1, max=25)
        pars['sigma2'].set(value=5, min=1, max=12)
        out = mod.fit(y, pars, x=x)
        #print(out.fit_report())
        ps = get_Mod_paramsValues(out)
        A, m1, s1, m2, s2 = ps.val[0], ps.val[1], ps.val[2], ps.val[3], ps.val[4]  
        # produces dataset for r-squared
        Aa1 = (x - m1)/s1
        Aaa1 = []
        for i in range(len(Aa1)):
                Aaa1.append(math.erf(Aa1.iloc[i]))
        Aa2 = -(x - m2)/s2
        Aaa2 = []
        for i in range(len(Aa2)):
                Aaa2.append(math.erf(Aa2.iloc[i]))
        Ayy = []
        for i in range(len(Aaa1)):
                Ayy.append((A/2)* (Aaa1[i] + Aaa2[i]))

        # Here the residual of the dataset is calculated
        resid = (y - Ayy)

        return resid, ps, rang

def point_resid(x, y, a, c1, s1, c2, s2):
    Aa1 = (x - c1)/s1
    Aaa1 = math.erf(Aa1)
    Aa2 = -(x - c2)/s2
    Aaa2 = math.erf(Aa2)
    Ayy = (a/2)* (Aaa1 + Aaa2)
    # Here the residual of the dataset is calculated
    resid = (y - Ayy)
    return resid
# %%
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
# %%
#Friedman
low_mon_num = np.histogram_bin_edges(clim_locs_historic.iloc[0,:], bins='fd')
low_mon_mean = np.histogram_bin_edges(clim_locs_historic.iloc[1,:], bins='fd')
low_mon_low = np.histogram_bin_edges(clim_locs_historic.iloc[2,:], bins='fd')
high_mon_num = np.histogram_bin_edges(clim_locs_historic.iloc[3,:], bins='fd')
high_mon_mean = np.histogram_bin_edges(clim_locs_historic.iloc[4,:], bins='fd')
high_mon_upp = np.histogram_bin_edges(clim_locs_historic.iloc[5,:], bins='fd')
clim_allmean = np.histogram_bin_edges(clim_locs_historic.iloc[6,:], bins='fd')

len_of_bins = [len(low_mon_num) -1, len(low_mon_mean) -1, len(low_mon_low) -1,
               len(high_mon_num) -1, len(high_mon_mean) -1, len(high_mon_upp)-1, len(clim_allmean) -1]

rank_clim_locs = np.zeros((7,35))
for i in range(0,35):
    for j in range(0, len(low_mon_num)-1):
#        if clim_locs_historic.iloc[0,j] in range(low_mon_num[j], low_mon_num[j+1]):
#            rank_clim_locs[0,j] = j
        if low_mon_num[j] <= clim_locs_historic.iloc[0,i] <= low_mon_num[j+1]:
            print(i,j)
            rank_clim_locs[0,i] = j
    for j in range(0,len(low_mon_mean)-1):
        if low_mon_mean[j] <= clim_locs_historic.iloc[1,i] <= low_mon_mean[j+1]:
            print(i,j)
            rank_clim_locs[1,i] = j
    for j in range(0,len(low_mon_low)-1):
        if low_mon_low[j] <= clim_locs_historic.iloc[2,i] <= low_mon_low[j+1]:
            print(i,j)
            rank_clim_locs[2,i] = j
    for j in range(0,len(high_mon_num)-1):
        if high_mon_num[j] <= clim_locs_historic.iloc[3,i] <= high_mon_num[j+1]:
            print(i,j)
            rank_clim_locs[3,i] = j
    for j in range(0,len(high_mon_mean)-1):
        if high_mon_mean[j] <= clim_locs_historic.iloc[4,i] <= high_mon_mean[j+1]:
            print(i,j)
            rank_clim_locs[4,i] = j               
    for j in range(0,len(high_mon_upp)-1):
        if high_mon_upp[j] <= clim_locs_historic.iloc[5,i] <= high_mon_upp[j+1]:
            print(i,j)
            rank_clim_locs[5,i] = j     
    for j in range(0,len(clim_allmean)-1):
        if clim_allmean[j] <= clim_locs_historic.iloc[6,i] <= clim_allmean[j+1]:
            print(i,j)
            rank_clim_locs[6,i] = j     

# %%
# Method building for reducing PFT impact on residual analysis
param_span = np.zeros((5,16))
for i in range(1, 17):
    dataset = PSIIContr[PSIIContr['Adjusted PFT'] == i]
    Ordered = dataset.sort_values(by='HeatMid')
    x = Ordered['HeatMid']
    x0 = x.iloc[:]
    y = Ordered['phiPSIImax']
    y0 = y.iloc[:]
    # Quad Model run
    mod = RectangleModel(form='erf')
    mod.set_param_hint('amplitude', value=0.8, min=0.75, max=0.9)
    mod.set_param_hint('center1', value=-3, min=-15, max=10)
    mod.set_param_hint('center2', value=45, min=15, max=60)
    mod.set_param_hint('sigma1', value=7, min=1, max=12)
    mod.set_param_hint('sigma2', value=7, min=1, max=12)
    pars = mod.guess(y0, x=x0)
    pars['amplitude'].set(value=0.8, min=0.6, max=0.83)
    pars['center1'].set(value=-6, min=-23, max=7)
    pars['center2'].set(value=46, min=35, max=57)
    pars['sigma1'].set(value=7, min=1, max=25)
    pars['sigma2'].set(value=5, min=1, max=12)
    out = mod.fit(y, pars, x=x)
    ps = get_Mod_paramsValues(out)
    print(ps['val'][:])
    param_span[:,i-1] = ps['val'][:]
# %%
def removed_PFT_resid(dataset):
    datasetx = dataset['HeatMid']
    datasety = dataset['phiPSIImax']
    new_resid = np.zeros((len(dataset), 4))
    for i in range(0,len(datasetx)):
        pft = int(dataset['Adjusted PFT'].iloc[i])
        new_resid[i,0] = datasetx.iloc[i]
        new_resid[i,1] = datasety.iloc[i]
        new_resid[i,2] = pft
        new_resid[i,3] = point_resid(datasetx.iloc[i], datasety.iloc[i],
                                     param_span[0,pft-1], param_span[1,pft-1], param_span[2,pft-1],
                                     param_span[3,pft-1], param_span[4,pft-1])
    return new_resid
# %%
all_try_one = removed_PFT_resid(PSIIContr)
# %%
# Create the residual column based on all PSIIGEO data
# Also builds in the columns in an oraginzed by HeatMid for the ANOVA cates
# (need to make alternate group, mischaracterization of original attempt)
# (to recreate original set, sub back in the # items)
PSIIClim = PSIIGEO[PSIIGEO['Climate'] == 1]
Ordered_set = PSIIClim.sort_values(by='HeatMid') #PSIIGEO.sort_values(by='HeatMid')
# Ordered_set['residual'] = get_set_resid(PSIIClim) #get_set_resid(PSIIGEO)
Ordered_set['residual'] = removed_PFT_resid(PSIIClim)[:,3]
Ordered_set['loc num'] = Ordered_set['residual']
Ordered_set['LTMon'] = Ordered_set['residual']
Ordered_set['LTMean'] = Ordered_set['residual']
Ordered_set['LTExt'] = Ordered_set['residual']
Ordered_set['HTMon'] = Ordered_set['residual']
Ordered_set['HTMean'] = Ordered_set['residual']
Ordered_set['HTExt'] = Ordered_set['residual']
Ordered_set['CLMean'] = Ordered_set['residual']
for i in range(0,len(Ordered_set)):
    for j in range(0,35):
        if Ordered_set['latlon'].iloc[i] == locations[j]:
            Ordered_set['loc num'].iloc[i] = j
            Ordered_set['LTMon'].iloc[i] = rank_clim_locs[0,j]
            Ordered_set['LTMean'].iloc[i] = rank_clim_locs[1,j]
            Ordered_set['LTExt'].iloc[i] = rank_clim_locs[2,j]
            Ordered_set['HTMon'].iloc[i] = rank_clim_locs[3,j]
            Ordered_set['HTMean'].iloc[i] = rank_clim_locs[4,j]
            Ordered_set['HTExt'].iloc[i] = rank_clim_locs[5,j]
            Ordered_set['CLMean'].iloc[i] = rank_clim_locs[6,j]

# %%
# Generate a method for making the ranked column values for any ANOVA combo


# Need to check for the 43 term below if that is consistent with raw run of code.


def ANOVA_stuff(dataset, i, j, k):
    """Dataset in this case should always be PSIIGEO or
    Ordered_set. i, j, k are the classifying numbers
    of the chosen 'axes' acourding to the order they appear
    in len_of_bins. Assumes all the column data for the
    selected i,j,k are fully filled in and residual column 
    is already generated. To make the column index call align,
    make sure to follow the above script to generate the
    right number of columns"""
    PSII3ANOVA = dataset
    OneXtwo = np.zeros([len_of_bins[i], len_of_bins[j]])
    for n in range(0,len_of_bins[i]):
        for m in range(0,len_of_bins[j]):
            OneXtwo[n,m] = np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:, 43+i] == n) & (PSII3ANOVA.iloc[:,43+j] == m)])
    OneXthree = np.zeros([len_of_bins[i], len_of_bins[k]])
    for n in range(0,len_of_bins[i]):
        for m in range(0,len_of_bins[k]):
            OneXthree[n,m] = np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:, 43+i] == n) & (PSII3ANOVA.iloc[:,43+k] == m)])
    TwoXthree = np.zeros([len_of_bins[j], len_of_bins[k]])
    for n in range(0,len_of_bins[j]):
        for m in range(0,len_of_bins[k]):
            TwoXthree[n,m] = np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:, 43+j] == n) & (PSII3ANOVA.iloc[:,43+k] == m)])
    OneSolo = np.zeros(len_of_bins[i])
    for n in range(0, len_of_bins[i]):
        OneSolo[n] = np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:,43+i] == n)])
    TwoSolo = np.zeros(len_of_bins[j])
    for n in range(0, len_of_bins[j]):
        TwoSolo[n] = np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:,43+j] == n)])
    ThreeSolo = np.zeros(len_of_bins[k])
    for n in range(0,len_of_bins[k]):
        ThreeSolo[n] = np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:,43+k] == n)])
    mu = np.mean(PSII3ANOVA['residual'])

    PSII3ANOVA['ART T way'] = PSII3ANOVA['phiPSIImax']
    PSII3ANOVA['ART t way'] = PSII3ANOVA['phiPSIImax']
    PSII3ANOVA['ART S way'] = PSII3ANOVA['phiPSIImax']
    PSII3ANOVA['ART Tt way'] = PSII3ANOVA['phiPSIImax']
    PSII3ANOVA['ART TS way'] = PSII3ANOVA['phiPSIImax']
    PSII3ANOVA['ART tS way'] = PSII3ANOVA['phiPSIImax']
    PSII3ANOVA['ART TtS way'] = PSII3ANOVA['phiPSIImax']

    for x in range(len(PSII3ANOVA['phiPSIImax'])):
        #print(x)
        n = int(PSII3ANOVA.iloc[x, 43+i])
        #print(i)
        m = int(PSII3ANOVA.iloc[x, 43+j])
        #print(j)
        p = int(PSII3ANOVA.iloc[x, 43+k])
        #print(k)
        PSII3ANOVA['ART T way'].iloc[x] = (PSII3ANOVA['residual'].iloc[x] - np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:, 43+i] == n) & (PSII3ANOVA.iloc[:, 43+j] == m) & (PSII3ANOVA.iloc[:, 43+k] == p)]) + OneSolo[n] - mu)
        PSII3ANOVA['ART t way'].iloc[x] = (PSII3ANOVA['residual'].iloc[x] - np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:,43+i] == n) & (PSII3ANOVA.iloc[:, 43+j] == m) & (PSII3ANOVA.iloc[:, 43+k] == p)]) + TwoSolo[m] - mu)
        PSII3ANOVA['ART S way'].iloc[x] = (PSII3ANOVA['residual'].iloc[x] - np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:,43+i] == n) & (PSII3ANOVA.iloc[:, 43+j] == m) & (PSII3ANOVA.iloc[:, 43+k] == p)]) + ThreeSolo[p] - mu)
        PSII3ANOVA['ART Tt way'].iloc[x] = (PSII3ANOVA['residual'].iloc[x] - np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:,43+i] == n) & (PSII3ANOVA.iloc[:, 43+j] == m) & (PSII3ANOVA.iloc[:, 43+k] == p)]) + OneXtwo[n,m] - OneSolo[n] - TwoSolo[m] + mu)
        PSII3ANOVA['ART TS way'].iloc[x] = (PSII3ANOVA['residual'].iloc[x] - np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:,43+i] == n) & (PSII3ANOVA.iloc[:, 43+j] == m) & (PSII3ANOVA.iloc[:, 43+k] == p)]) + OneXthree[n,p] - OneSolo[n] - ThreeSolo[p] + mu)
        PSII3ANOVA['ART tS way'].iloc[x] = (PSII3ANOVA['residual'].iloc[x] - np.mean(PSII3ANOVA['residual'][(PSII3ANOVA.iloc[:,43+i] == n) & (PSII3ANOVA.iloc[:, 43+j] == m) & (PSII3ANOVA.iloc[:, 43+k] == p)]) + TwoXthree[m,p] - TwoSolo[m] - ThreeSolo[p] + mu)
        PSII3ANOVA['ART TtS way'].iloc[x] = (PSII3ANOVA['residual'].iloc[x] - OneXtwo[n,m] - OneXthree[n,p] - TwoXthree[m,p] + OneSolo[n] + TwoSolo[m] + ThreeSolo[p] - mu)

    return PSII3ANOVA

# %%
# Not sure if that function will work, guess we will see!
Attempt = ANOVA_stuff(Ordered_set, 1, 4, 6)

ART3way1 = Attempt.sort_values(by='ART T way')
ART3way1['number'] = ART3way1['ART T way'].rank()
ART3way2 = Attempt.sort_values(by='ART t way')
ART3way2['number'] = ART3way2['ART t way'].rank()
ART3way3 = Attempt.sort_values(by='ART S way')
ART3way3['number'] = ART3way3['ART S way'].rank()
ART3way_12 = Attempt.sort_values(by='ART Tt way')
ART3way_12['number'] = ART3way_12['ART Tt way'].rank()
ART3way_13 = Attempt.sort_values(by='ART TS way') 
ART3way_13['number'] = ART3way_13['ART TS way'].rank()
ART3way_23 = Attempt.sort_values(by='ART tS way')
ART3way_23['number'] = ART3way_23['ART tS way'].rank()
ART3way_all = Attempt.sort_values(by='ART TtS way')
ART3way_all['number'] = ART3way_all['ART TtS way'].rank()

# %%
# Now run thru the list of above dataframes adjusting Pogg
fir = 1
sec = 4
thr = 6
Pogg = ART3way_all
dCont = {'rank': Pogg['number'], 'One': Pogg.iloc[:, 43+fir], 'Two': Pogg.iloc[:, 43+sec], 'Three': Pogg.iloc[:, 43+thr]}
DfCont = pd.DataFrame(data=dCont)
model = ols('rank ~ C(One) + C(Two) + C(Three) + C(One):C(Two) + C(One):C(Three) + C(Two):C(Three) + C(One):C(Two):C(Three) -1', data=DfCont).fit()
anova_table = sm.stats.anova_lm(model, typ=3)
anova_table


 # %%
chooz = 100
itt = 600
set_choice = Ordered_set

point_spread = np.zeros((4,itt))
#for i in range(0,itt):
#    rand_set =

# %%
# used Ordered_set for this section
# Adds actual values based on location to allow exclusion methods
Ordered_set['LTval'] = Ordered_set['LTMean']
Ordered_set['HTval'] = Ordered_set['HTMean']
Ordered_set['CLval'] = Ordered_set['CLMean']
for i in range (0, len(Ordered_set['HTExt'])):
    j = Ordered_set['loc num'].iloc[i]
    Ordered_set['LTval'].iloc[i] = clim_locs_historic.iloc[1, int(j)]
    Ordered_set['HTval'].iloc[i] = clim_locs_historic.iloc[4, int(j)]
    Ordered_set['CLval'].iloc[i] = clim_locs_historic.iloc[6, int(j)]

ltm_min = clim_locs_historic.iloc[1,:].min()
ltm_max = clim_locs_historic.iloc[1,:].max()
htm_min = clim_locs_historic.iloc[4,:].min()
htm_max = clim_locs_historic.iloc[4,:].max()
clm_min = clim_locs_historic.iloc[6,:].min()
clm_max = clim_locs_historic.iloc[6,:].max()

# %%
def ran_method(dataset, num):
    ltmean_set = np.zeros((10,num))
    htmean_set = np.zeros((10,num))
    clmean_set = np.zeros((10,num))
    
    step_l = (ltm_max-ltm_min)/(100)
    step_h = (htm_max-htm_min)/(100)
    step_c = (clm_max-clm_min)/(100)

    for i in range(0,num):
        f1, g1 = random.sample(list(np.arange(ltm_min, ltm_max, step_l)), 2)
        a1, b1 = np.min([f1,g1]), np.max([f1,g1])
        f2, g2 = random.sample(list(np.arange(htm_min, htm_max, step_h)), 2)
        a2, b2 = np.min([f2,g2]), np.max([f2,g2])
        f3, g3 = random.sample(list(np.arange(clm_min, clm_max, step_c)), 2)
        a3, b3 = np.min([f3,g3]), np.max([f3,g3])

        new_set1 = dataset[(dataset['LTval'] > a1) & (dataset['LTval'] < b1)]
        new_set2 = dataset[(dataset['HTval'] > a2) & (dataset['HTval'] < b2)]
        new_set3 = dataset[(dataset['CLval'] > a3) & (dataset['CLval'] < b3)]

        n1 = len(new_set1['HeatMid'])
        n2 = len(new_set2['HeatMid'])
        n3 = len(new_set3['HeatMid'])

        ltmean_set[0,i] = a1
        ltmean_set[1,i] = b1
        ltmean_set[2,i] = n1
        ltmean_set[3,i] = np.min(new_set1['HeatMid'])
        ltmean_set[4,i] = np.max(new_set1['HeatMid'])
        if n1 > 30:
            ltmean_set[5:,i] = get_set_resid(new_set1)[1]['val'][:]
        else:
            ltmean_set[5:,i] = -999

        htmean_set[0,i] = a2
        htmean_set[1,i] = b2
        htmean_set[2,i] = n2
        htmean_set[3,i] = np.min(new_set2['HeatMid'])
        htmean_set[4,i] = np.max(new_set2['HeatMid'])
        if n2 > 30:
            htmean_set[5:,i] = get_set_resid(new_set2)[1]['val'][:]
        else:
            htmean_set[5:,i] = -999

        clmean_set[0,i] = a3
        clmean_set[1,i] = b3
        clmean_set[2,i] = n3
        clmean_set[3,i] = np.min(new_set3['HeatMid'])
        clmean_set[4,i] = np.max(new_set3['HeatMid'])
        if n3 > 30:
            clmean_set[5:,i] = get_set_resid(new_set3)[1]['val'][:]
        else:
            clmean_set[5:,i] = -999


    return ltmean_set, htmean_set, clmean_set


def shrink_method(dataset, num):
    ltmean_set = np.zeros((9,num))
    htmean_set = np.zeros((9,num))
    
    step_l = (ltm_max-ltm_min)/(num)
    step_h = (htm_max-htm_min)/(num)

    for i in range(0,num):
        a1 = ltm_min + i*step_l
        a2 = htm_max - i*step_h

        new_set1 = dataset[dataset['LTval'] > a1]
        new_set2 = dataset[dataset['HTval'] < a2]

        n1 = len(new_set1['HeatMid'])
        n2 = len(new_set2['HeatMid'])

        ltmean_set[0,i] = a1
        ltmean_set[1,i] = n1
        ltmean_set[2,i] = np.min(new_set1['HeatMid'])
        ltmean_set[3,i] = np.max(new_set1['HeatMid'])
        if n1 > 30:
            ltmean_set[4:,i] = get_set_resid(new_set1)[1]['val'][:]
        else:
            ltmean_set[4:,i] = -999

        htmean_set[0,i] = a2
        htmean_set[1,i] = n2
        htmean_set[2,i] = np.min(new_set2['HeatMid'])
        htmean_set[3,i] = np.max(new_set2['HeatMid'])
        if n1 > 30:
            htmean_set[4:,i] = get_set_resid(new_set2)[1]['val'][:]
        else:
            htmean_set[4:,i] = -999


    return ltmean_set, htmean_set


def quant_method(dataset, num):
    return 47


# %%
try1_1, try1_2, try1_3 = ran_method(Ordered_set, 1000)
# method for plotting parameters to remove the bad -999 runs
plt.hist(np.where(try1_1[5,:]==-999, np.nan, try1_1[5,:]))
# %%
try2_1, try2_2 = shrink_method(Ordered_set, 50)

