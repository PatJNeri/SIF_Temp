# ClimatologyStats
# Patrick Neri
# LETS DO THIS
# hastools2
# %%
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
import seaborn as sns
import cartopy.crs as ccrs
from cartopy import feature as cfeature

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
        #mod.set_param_hint('amplitude', value=0.8, min=0.75, max=0.9)
        #mod.set_param_hint('center1', value=-3, min=-15, max=10)
        #mod.set_param_hint('center2', value=45, min=15, max=60)
        #mod.set_param_hint('sigma1', value=7, min=1, max=12)
        #mod.set_param_hint('sigma2', value=7, min=1, max=12)
        pars = mod.guess(y0, x=x0)
        pars['amplitude'].set(value=0.8, min=0.6, max=0.83)
        pars['center1'].set(value=-3, min=-23, max=15)
        pars['center2'].set(value=46, min=35, max=57)
        pars['sigma1'].set(value=7, min=1, max=25)
        pars['sigma2'].set(value=5, min=1, max=12)
        out = mod.fit(y, pars, x=x)
        #print(out.fit_report())
        ps = get_Mod_paramsValues(out)
        A, m1, s1, m2, s2 = ps.val[0], ps.val[1], ps.val[2], ps.val[3], ps.val[4] 
        gold = [(m1 + 2*s1), (m2 - 2*s2)] 
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
        correlation_matrix = np.corrcoef(y, Ayy)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        return resid, r_squared, ps, gold, rang

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
#Precip = pd.read_csv('Hist_Precip.csv')
#Qbot = pd.read_csv('Hist_Qbot.csv')
#Solar = pd.read_csv('Hist_Solar.csv')
TempB = pd.read_csv('Hist_TempB.csv')

# Convert to DataFrame
#P = pd.DataFrame(data=Precip)
#Q = pd.DataFrame(data=Qbot)
#S = pd.DataFrame(data=Solar)
T = pd.DataFrame(data=TempB)
# %%
# Need to form a way to convert form the string date column to a more
# usabe datetime method for ease of call and averaging.
#P['time'] = pd.to_datetime(P['time'])
#Q['time'] = pd.to_datetime(Q['time'])
#S['time'] = pd.to_datetime(S['time'])
T['time'] = pd.to_datetime(T['time'])

month_temp_try = T.groupby(T['time'].dt.month).mean()

# %%
# methods of saving them (building a loop)
clim_locs_historic = pd.DataFrame()
for j in range(0,35):
    location = 'location ' + str(j)
    #print(location)
    ranked_3mon_sum = np.zeros((12, 1))
    ranked_3mon_sum[0] = (np.mean((month_temp_try[location][1], month_temp_try[location][2], month_temp_try[location][12])))
    for i in range(2,12):
        ranked_3mon_sum[i-1] = (np.mean(month_temp_try[location][i-1:i+1]))
    ranked_3mon_sum[11] = (np.mean((month_temp_try[location][11], month_temp_try[location][12], month_temp_try[location][1])))
    # Finding max 3-month running average to define 'season'
    r_try = np.append(ranked_3mon_sum, np.arange(1,13).reshape((12,1)))

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
            #print(i,j)
            rank_clim_locs[0,i] = j
    for j in range(0,len(low_mon_mean)-1):
        if low_mon_mean[j] <= clim_locs_historic.iloc[1,i] <= low_mon_mean[j+1]:
            #print(i,j)
            rank_clim_locs[1,i] = j
    for j in range(0,len(low_mon_low)-1):
        if low_mon_low[j] <= clim_locs_historic.iloc[2,i] <= low_mon_low[j+1]:
            #print(i,j)
            rank_clim_locs[2,i] = j
    for j in range(0,len(high_mon_num)-1):
        if high_mon_num[j] <= clim_locs_historic.iloc[3,i] <= high_mon_num[j+1]:
            #print(i,j)
            rank_clim_locs[3,i] = j
    for j in range(0,len(high_mon_mean)-1):
        if high_mon_mean[j] <= clim_locs_historic.iloc[4,i] <= high_mon_mean[j+1]:
            #print(i,j)
            rank_clim_locs[4,i] = j               
    for j in range(0,len(high_mon_upp)-1):
        if high_mon_upp[j] <= clim_locs_historic.iloc[5,i] <= high_mon_upp[j+1]:
            #print(i,j)
            rank_clim_locs[5,i] = j     
    for j in range(0,len(clim_allmean)-1):
        if clim_allmean[j] <= clim_locs_historic.iloc[6,i] <= clim_allmean[j+1]:
            #print(i,j)
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
    #print(ps['val'][:])
    param_span[:,i-1] = ps['val'][:]

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
            ltmean_set[5:,i] = get_set_resid(new_set1)[2]['val'][:]
        else:
            ltmean_set[5:,i] = -999

        htmean_set[0,i] = a2
        htmean_set[1,i] = b2
        htmean_set[2,i] = n2
        htmean_set[3,i] = np.min(new_set2['HeatMid'])
        htmean_set[4,i] = np.max(new_set2['HeatMid'])
        if n2 > 30:
            htmean_set[5:,i] = get_set_resid(new_set2)[2]['val'][:]
        else:
            htmean_set[5:,i] = -999

        clmean_set[0,i] = a3
        clmean_set[1,i] = b3
        clmean_set[2,i] = n3
        clmean_set[3,i] = np.min(new_set3['HeatMid'])
        clmean_set[4,i] = np.max(new_set3['HeatMid'])
        if n3 > 30:
            clmean_set[5:,i] = get_set_resid(new_set3)[2]['val'][:]
        else:
            clmean_set[5:,i] = -999


    return ltmean_set, htmean_set, clmean_set


def shrink_method(dataset, num):
    ltmean_set = np.zeros((9,num))
    htmean_set = np.zeros((9,num))
    
    step_l = (ltm_max-ltm_min)/(num)
    step_h = (htm_max-htm_min)/(num)

    for i in range(0,num):
        a1 = ltm_max - i*step_l
        a2 = htm_max - i*step_h

        new_set1 = dataset[dataset['LTval'] < a1]
        new_set2 = dataset[dataset['HTval'] < a2]

        n1 = len(new_set1['HeatMid'])
        n2 = len(new_set2['HeatMid'])

        ltmean_set[0,i] = a1
        ltmean_set[1,i] = n1
        ltmean_set[2,i] = np.min(new_set1['HeatMid'])
        ltmean_set[3,i] = np.max(new_set1['HeatMid'])
        if n1 > 30:
            ltmean_set[4:,i] = get_set_resid(new_set1)[2]['val'][:]
        else:
            ltmean_set[4:,i] = -999

        htmean_set[0,i] = a2
        htmean_set[1,i] = n2
        htmean_set[2,i] = np.min(new_set2['HeatMid'])
        htmean_set[3,i] = np.max(new_set2['HeatMid'])
        if n1 > 30:
            htmean_set[4:,i] = get_set_resid(new_set2)[2]['val'][:]
        else:
            htmean_set[4:,i] = -999


    return ltmean_set, htmean_set


def quant_method(dataset, percent):
    gap = int(100 - percent)

    ltmean_set = np.zeros((9,gap))
    htmean_set = np.zeros((9,gap))
    clmean_set = np.zeros((9,gap))

    for i in range(0, gap):
        a = i
        b = percent + i

        low_q1 = np.quantile(clim_locs_historic.iloc[1,:], a/100)
        high_q1 = np.quantile(clim_locs_historic.iloc[1,:], b/100)
        low_q2 = np.quantile(clim_locs_historic.iloc[4,:], a/100)
        high_q2 = np.quantile(clim_locs_historic.iloc[4,:], b/100)
        low_q3 = np.quantile(clim_locs_historic.iloc[6,:], a/100)
        high_q3 = np.quantile(clim_locs_historic.iloc[6,:], b/100)

        new_set1 = dataset[(dataset['LTval'] > low_q1) & (dataset['LTval'] < high_q1)]
        new_set2 = dataset[(dataset['HTval'] > low_q2) & (dataset['HTval'] < high_q2)]
        new_set3 = dataset[(dataset['CLval'] > low_q3) & (dataset['CLval'] < high_q3)]

        n1 = len(new_set1['HeatMid'])
        n2 = len(new_set2['HeatMid'])
        n3 = len(new_set3['HeatMid'])

        ltmean_set[0,i] = (a + b)/2
        ltmean_set[1,i] = n1
        ltmean_set[2,i] = np.min(new_set1['HeatMid'])
        ltmean_set[3,i] = np.max(new_set1['HeatMid'])
        if n1 > 30:
            ltmean_set[4:,i] = get_set_resid(new_set1)[2]['val'][:]
        else:
            ltmean_set[4:,i] = -999

        htmean_set[0,i] = (a + b)/2
        htmean_set[1,i] = n2
        htmean_set[2,i] = np.min(new_set2['HeatMid'])
        htmean_set[3,i] = np.max(new_set2['HeatMid'])
        if n2 > 30:
            htmean_set[4:,i] = get_set_resid(new_set2)[2]['val'][:]
        else:
            htmean_set[4:,i] = -999

        clmean_set[0,i] = (a + b)/2
        clmean_set[1,i] = n3
        clmean_set[2,i] = np.min(new_set3['HeatMid'])
        clmean_set[3,i] = np.max(new_set3['HeatMid'])
        if n3 > 30:
            clmean_set[4:,i] = get_set_resid(new_set3)[2]['val'][:]
        else:
            clmean_set[4:,i] = -999        

    return ltmean_set, htmean_set, clmean_set
#####################################################################


# RUN ABOVE HERE FOR EASE OF CODING


#####################################################################
    
# %%
try1_1, try1_2, try1_3 = ran_method(Ordered_set, 500)
# method for plotting parameters to remove the bad -999 runs
plt.hist(np.where(try1_1[5,:]==-999, np.nan, try1_1[5,:]))

# %%
a = [try1_1[5,:] == -999]
b = [try1_2[5,:] == -999]
c = [try1_3[5,:] == -999]
dry1_1 = np.ma.masked_array(try1_1, mask=[a,a,a,a,a,a,a,a,a,a])
dry1_2 = np.ma.masked_array(try1_2, mask=[b,b,b,b,b,b,b,b,b,b])
dry1_3 = np.ma.masked_array(try1_3, mask=[c,c,c,c,c,c,c,c,c,c])

# %%
x = dry1_3[2,:]
y = dry1_3[8,:]
z = dry1_3[9,:]

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x,y,z, marker='o')

# %%
fig, ax = plt.subplots()
gap = ax.scatter(y,z, c=x, cmap='jet')
d = plt.colorbar(gap)
ax.set_xlabel('y')
ax.set_ylabel('z')
d.set_label('x')
# %%
try2_1, try2_2 = shrink_method(Ordered_set, 50)

aa = [try2_1[5,:] == -999]
bb = [try2_2[5,:] == -999]
dry2_1 = np.ma.masked_array(try2_1, mask=[aa,aa,aa,aa,aa,aa,aa,aa,aa])
dry2_2 = np.ma.masked_array(try2_2, mask=[bb,bb,bb,bb,bb,bb,bb,bb,bb])


# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dry2_1[5,:40],dry2_1[6,:40],dry2_1[1,:40], marker='o')

# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(0,90) #adjusts the view (elevation and azimuth angles)
x = np.linspace(-32, 63, 400)
for i in range(0,45):
    z_set = np.full(400, 2*i)
    if dry2_1[5,i] != np.nan:
        a, m1, s1, m2, s2 = dry2_1[4:, i]
        Aa1 = (x - m1)/s1
        Aaa1 = []
        for i in range(len(Aa1)):
                Aaa1.append(math.erf(Aa1[i]))
        Aa2 = -(x - m2)/s2
        Aaa2 = []
        for i in range(len(Aa2)):
                Aaa2.append(math.erf(Aa2[i]))
        Ayy = []
        for i in range(len(Aaa1)):
                Ayy.append((a/2)* (Aaa1[i] + Aaa2[i]))
        ax.plot(x, z_set, Ayy)
# %%
fig = plt.figure(figsize=(8,8))
ax = plt.axes()
x = np.linspace(-32, 63, 400)
for i in [0,10,20,30,40]:
    name = str(i)
    name2 = round(dry2_1[0,i], 2)
    print(name2)
    if dry2_1[5,i] != np.nan:
        a, m1, s1, m2, s2 = dry2_1[4:, i]
        Aa1 = (x - m1)/s1
        Aaa1 = []
        for i in range(len(Aa1)):
                Aaa1.append(math.erf(Aa1[i]))
        Aa2 = -(x - m2)/s2
        Aaa2 = []
        for i in range(len(Aa2)):
                Aaa2.append(math.erf(Aa2[i]))
        Ayy = []
        for i in range(len(Aaa1)):
                Ayy.append((a/2)* (Aaa1[i] + Aaa2[i]))
        ax.plot(x, Ayy, label=str(name2))
ax.grid(True)
ax.set_ylim((0,0.85))
ax.set_xlabel('Experiment Measured Temperature (C)')
ax.set_ylabel('Maximum Quantum Efficiency of PSII')
ax.set_title('Selected Models based on HTMean subset')
ax.legend(title='Avg. winter max. temp. (K)')

# %%
# improve by making wireframe method
# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
z_set = np.zeros((401, 50))
x = np.linspace(-35, 65, 401)

for j in range(0,50):
    a, m1, s1, m2, s2 = dry2_1[4:, j] # this part to change
    Aa1 = (x - m1)/s1
    Aaa1 = []
    for i in range(len(Aa1)):
            Aaa1.append(math.erf(Aa1[i]))
    Aa2 = -(x - m2)/s2
    Aaa2 = []
    for i in range(len(Aa2)):
            Aaa2.append(math.erf(Aa2[i]))
    Ayy = []
    for i in range(len(Aaa1)):
        Ayy.append((a/2)* (Aaa1[i] + Aaa2[i]))
    z_set[:,j] = Ayy

fig, ax = plt.subplots()
gap = ax.pcolor(z_set, cmap='jet')
ax.set_title('dry2_1 color scheme shrink')
ax.set_yticklabels(x[::50])
ax.set_xticklabels(dry2_1[0,::10], rotation=60)
d = plt.colorbar(gap)

# %%
q = 33
try3_1, try3_2, try3_3 = quant_method(Ordered_set, q)


plt.plot(try3_1[0,:], try3_1[3,:], label='try3_1')
plt.plot(try3_2[0,:], try3_2[3,:], label='try3_2')
plt.plot(try3_3[0,:], try3_3[3,:], label='try3_3')
plt.title('Axis # of points by Quantile (20)')
plt.xlabel('Quantile range center')
plt.ylabel('# of points')
plt.legend()

z_set = np.zeros((401, 100-q))
x = np.linspace(-35, 65, 401)
# %%
plt.plot(try3_1[0,:], try3_1[5,:], '-g')
plt.plot(try3_2[0,:], try3_2[5,:], '-r')
plt.plot(try3_3[0,:], try3_3[5,:], '-b')
plt.plot(try3_1[0,:], try3_1[7,:], '-g')
plt.plot(try3_2[0,:], try3_2[7,:], '-r')
plt.plot(try3_3[0,:], try3_3[7,:], '-b')
plt.fill_between(try3_1[0,:], try3_1[7,:], try3_1[5,:], alpha=0.3, color='g', )
plt.fill_between(try3_2[0,:], try3_2[7,:], try3_2[5,:], alpha=0.3, color='r')
plt.fill_between(try3_3[0,:], try3_3[7,:], try3_3[5,:], alpha=0.3, color='b')
plt.ylim((-40, 65))

#%%
for j in range(0,100-q):
    a, m1, s1, m2, s2 = try3_2[4:, j] # this part to change
    Aa1 = (x - m1)/s1
    Aaa1 = []
    for i in range(len(Aa1)):
            Aaa1.append(math.erf(Aa1[i]))
    Aa2 = -(x - m2)/s2
    Aaa2 = []
    for i in range(len(Aa2)):
            Aaa2.append(math.erf(Aa2[i]))
    Ayy = []
    for i in range(len(Aaa1)):
        Ayy.append((a/2)* (Aaa1[i] + Aaa2[i]))
    z_set[:,j] = Ayy

fig, ax = plt.subplots()
gap = ax.pcolor(z_set, cmap='jet')
ax.set_title('try3_1 color scheme quantile 25')
ax.set_yticklabels(x[::50])
#ax.set_xticklabels(try3_1[0,::10], rotation=60)
d = plt.colorbar(gap)
# %%
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8,10))
for q in [75, 66, 50, 33, 25]:
    try31, try32, try33 = quant_method(Ordered_set, q)

    a = ax[0,0].scatter(try31[5,:] + 2*try31[6,:], try31[0,:], c=try31[2,:], cmap='jet')

    b = ax[0,1].scatter(try31[7,:] - 2*try31[8,:], try31[0,:], c=try31[3,:], cmap='jet')
    ax[0,0].set_xlim((-25, 35))
    ax[0,1].set_xlim((20, 57))
    
    c = ax[1,0].scatter(try32[5,:] + 2*try32[6,:], try32[0,:], c=try32[2,:], cmap='jet')
    
    d = ax[1,1].scatter(try32[7,:] - 2*try32[8,:], try32[0,:], c=try32[3,:], cmap='jet')
    ax[1,0].set_xlim((-25, 35))
    ax[1,1].set_xlim((20, 57))
    
    e = ax[2,0].scatter(try33[5,:] + 2*try33[6,:], try33[0,:], c=try33[2,:], cmap='jet')
    
    f = ax[2,1].scatter(try33[7,:] - 2*try33[8,:], try33[0,:], c=try33[3,:], cmap='jet')
    ax[2,0].set_xlim((-25, 35))
    ax[2,1].set_xlim((20, 57))
    
plt.colorbar(a, ax=ax[0,0])
ax[0,0].set_yticklabels((-41.8,-10.4,4.6,16,21.4))
plt.colorbar(b, ax=ax[0,1])
ax[0,1].set_yticklabels((-41.8,-10.4,4.6,16,21.4))
plt.colorbar(c, ax=ax[1,0])
ax[1,0].set_yticklabels((6.2, 11.1, 17.4, 23.3, 27.1))
plt.colorbar(d, ax=ax[1,1])
ax[1,1].set_yticklabels((6.2, 11.1, 17.4, 23.3, 27.1))
plt.colorbar(e, ax=ax[2,0])
ax[2,0].set_yticklabels((-14.5, 2.4, 10.1, 20.2, 24.2))
plt.colorbar(f, ax=ax[2,1])
ax[2,1].set_yticklabels((-14.5, 2.4, 10.1, 20.2, 24.2))
# %%
# Processing modelling step
q_25 = quant_method(Ordered_set, 25)[0]
setq_25 = pd.DataFrame(data=q_25.T, columns=['center', 'number', 'low', 'high',
                                           'a', 'c1', 's1', 'c2', 's2'])
q_33 = quant_method(Ordered_set, 33)[0]
setq_33 = pd.DataFrame(data=q_33.T, columns=['center', 'number', 'low', 'high',
                                           'a', 'c1', 's1', 'c2', 's2'])
q_40 = quant_method(Ordered_set, 40)[0]
setq_40 = pd.DataFrame(data=q_40.T, columns=['center', 'number', 'low', 'high',
                                           'a', 'c1', 's1', 'c2', 's2'])
q_50 = quant_method(Ordered_set, 50)[0]
setq_50 = pd.DataFrame(data=q_50.T, columns=['center', 'number', 'low', 'high',
                                           'a', 'c1', 's1', 'c2', 's2'])
q_60 = quant_method(Ordered_set, 60)[0]
setq_60 = pd.DataFrame(data=q_60.T, columns=['center', 'number', 'low', 'high',
                                           'a', 'c1', 's1', 'c2', 's2'])
q_66 = quant_method(Ordered_set, 66)[0]
setq_66 = pd.DataFrame(data=q_66.T, columns=['center', 'number', 'low', 'high',
                                           'a', 'c1', 's1', 'c2', 's2'])
q_75 = quant_method(Ordered_set, 75)[0]
setq_75 = pd.DataFrame(data=q_75.T, columns=['center', 'number', 'low', 'high',
                                           'a', 'c1', 's1', 'c2', 's2'])
q_80 = quant_method(Ordered_set, 80)[0]
setq_80 = pd.DataFrame(data=q_80.T, columns=['center', 'number', 'low', 'high',
                                           'a', 'c1', 's1', 'c2', 's2'])

# %%
# https://seaborn.pydata.org/tutorial/regression.html
sns.regplot(x='c2', y='s2', data=setq_25[setq_25['high'] > 45], robust=True)
#sns.lmplot(x='c2', y='s2', data=setq_25, hue='high')

# %%
#plt.scatter(setq_25['c1'], setq_25['s1'], label='25')
plt.scatter(setq_33['c1'], setq_33['s1'], label='33')
plt.scatter(setq_40['c1'], setq_40['s1'], label='40')
plt.scatter(setq_50['c1'], setq_50['s1'], label='50')
plt.scatter(setq_60['c1'], setq_60['s1'], label='60')
plt.scatter(setq_66['c1'], setq_66['s1'], label='66')
plt.scatter(setq_75['c1'], setq_75['s1'], label='75')
plt.scatter(setq_80['c1'], setq_80['s1'], label='80')
plt.legend()

# %%
# https://stackoverflow.com/questions/22852244/how-to-get-the-numerical-fitting-results-when-plotting-a-regression-in-seaborn
import statsmodels.api as sm


def simple_regplot(
    x, y, n_std=2, n_pts=100, ax=None, scatter_kws=None, line_kws=None, ci_kws=None
):
    """ Draw a regression line with error interval. """
    ax = plt.gca() if ax is None else ax

    # calculate best-fit line and interval
    x_fit = sm.add_constant(x)
    fit_results = sm.OLS(y, x_fit).fit()

    eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_pts))
    pred = fit_results.get_prediction(eval_x)

    # draw the fit line and error interval
    ci_kws = {} if ci_kws is None else ci_kws
    ax.fill_between(
        eval_x[:, 1],
        pred.predicted_mean - n_std * pred.se_mean,
        pred.predicted_mean + n_std * pred.se_mean,
        alpha=0.5,
        **ci_kws,
    )
    line_kws = {} if line_kws is None else line_kws
    h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kws)

    # draw the scatterplot
    scatter_kws = {} if scatter_kws is None else scatter_kws
    ax.scatter(x, y, c=h[0].get_color(), **scatter_kws)

    return fit_results
# %%
fplot = simple_regplot(setq_50[setq_50['low'] < -7]['center'], setq_50[setq_50['low'] < -7]['c1'])
print(fplot.summary())
# %%
set_of_set = [setq_25, setq_33, setq_40, setq_50, setq_60, setq_66, setq_75, setq_80]
set_of_label = ['25', '33', '40', '50', '66', '75', '80']
fig, ax = plt.subplots()
for graph in set_of_set:
    ax.plot(graph['low'], graph['center'])
# %%
for graph in set_of_set:
    fplot = simple_regplot(graph[graph['low'] < -7]['c1'], graph[graph['low'] < -7]['center'])
    print(fplot.summary())
# %%
gplot1 = setq_25[setq_25['low'] < -6].append(setq_33[setq_33['low'] < -6]).append(setq_50[setq_50['low'] < -6]).append(setq_66[setq_66['low'] < -6]).append(setq_75[setq_75['low'] < -6])
# %%
fplot2 = simple_regplot(gplot1['c1'], gplot1['s1'])
print(fplot2.summary())
# %%
#####################################################################

# Python Pie Chart

#####################################################################
labels = ['Winter mean temperature', 'Summer mean temperature', 'Annual mean temperature',
          'Winter x Summer', 'Winter x Annual', 'Summer x Annual', '3 Way interaction',
          'Residual']
sizes = [7.2, 1.7, 5.5, 8.2, 19.9, 7.6, 45.2, 4.7]
explode = (0,0,0,0,0,0,0,0.1)

fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%')
ax1.axis('equal')
plt.show()
# %%
# Assumes q = 33 scenerio
for j in range(0,67):
    a, m1, s1, m2, s2 = try3_1[4:, j] # this part to change
    Aa1 = (x - m1)/s1
    Aaa1 = []
    for i in range(len(Aa1)):
            Aaa1.append(math.erf(Aa1[i]))
    Aa2 = -(x - m2)/s2
    Aaa2 = []
    for i in range(len(Aa2)):
            Aaa2.append(math.erf(Aa2[i]))
    Ayy = []
    for i in range(len(Aaa1)):
        Ayy.append((a/2)* (Aaa1[i] + Aaa2[i]))
    z_set[:,j] = Ayy

fig, axs = plt.subplots(ncols=2, figsize=(9,4), sharex=True, tight_layout=True)
axs[0].set_title('Representative Selection of \nModel Trends')
axs[0].set_ylabel('Maximum Quantum \nEfficiency of PSII')
axs[0].set_xlabel('Experimental Temperature ' + u'\u2103')
axs[1].set_title('Climatology Trends across \nQuantile System Approach')
axs[1].set_ylabel('Winter Annual Mean \nTemperature (WAT) ' + u'\u2103')
axs[1].set_xlabel('Experimental Temperature ' + u'\u2103')
gap = axs[1].pcolor(z_set.T, cmap='jet')
axs[0].plot(z_set[:,0], label='WAT = -12.4', color='blue')
axs[0].plot(z_set[:,14], label='WAT = 1.7', color='cyan')
axs[0].plot(z_set[:,25], label='WAT = 4.6', color='green')
axs[0].plot(z_set[:,40], label='WAT = 14.7', color='yellow')
axs[0].plot(z_set[:,60], label='WAT 20.9', color='red')
axs[0].grid(True)
axs[0].legend()
axs[1].set_xticklabels(x[::100])
axs[1].set_yticklabels(np.round((np.round(np.quantile(clim_locs_historic.iloc[1,:], [.165, .265, .365, .465, .565, .665, .765]),2)-273.15),2))
axs[1].set_aspect(401/67)

plt.colorbar(gap, label='Maximum Quantum \nEfficiency of PSII')
fig.suptitle('Winter Climatology Trends', fontsize=25)
# %%
# Assumes q = 33 scenerio
for j in range(0,67):
    a, m1, s1, m2, s2 = try3_2[4:, j] # this part to change
    Aa1 = (x - m1)/s1
    Aaa1 = []
    for i in range(len(Aa1)):
            Aaa1.append(math.erf(Aa1[i]))
    Aa2 = -(x - m2)/s2
    Aaa2 = []
    for i in range(len(Aa2)):
            Aaa2.append(math.erf(Aa2[i]))
    Ayy = []
    for i in range(len(Aaa1)):
        Ayy.append((a/2)* (Aaa1[i] + Aaa2[i]))
    z_set[:,j] = Ayy

fig, axs = plt.subplots(ncols=2, figsize=(9,4), sharex=True, tight_layout=True)
axs[0].set_title('Representative Selection of \nModel Trends')
axs[0].set_ylabel('Maximum Quantum \nEfficiency of PSII')
axs[0].set_xlabel('Experimental Temperature ' + u'\u2103')
axs[1].set_title('Climatology Trends across \nQuantile System Approach')
axs[1].set_ylabel('Summer Annual Mean \nTemperature (SAT) ' + u'\u2103')
axs[1].set_xlabel('Experimental Temperature ' + u'\u2103')
gap = axs[1].pcolor(z_set.T, cmap='jet')
axs[0].plot(z_set[:,0], label='SAT = 10.8', color='blue')
axs[0].plot(z_set[:,14], label='SAT = 16.1', color='cyan')
axs[0].plot(z_set[:,25], label='SAT = 17.6', color='green')
axs[0].plot(z_set[:,40], label='SAT = 22.6', color='yellow')
axs[0].plot(z_set[:,60], label='SAT 26.7', color='red')
axs[0].grid(True)
axs[0].legend()
axs[1].set_xticklabels(x[::100])
axs[1].set_yticklabels(np.round((np.round(np.quantile(clim_locs_historic.iloc[4,:], [.165, .265, .365, .465, .565, .665, .765]),2)-273.15),2))
axs[1].set_aspect(401/67)

plt.colorbar(gap, label='Maximum Quantum \nEfficiency of PSII')
fig.suptitle('Summer Climatology Trends', fontsize=25)
# %%

#####################################################################

# RUN AGAIN STARTING HERE FOR EASE 

#####################################################################

import random

def carlo_stats(dataset):
    Ordered = dataset.sort_values(by='HeatMid')
    x = Ordered['HeatMid']
    x0 = x.iloc[:]
    y = Ordered['phiPSIImax']
    y0 = y.iloc[:]
    mod = RectangleModel(form='erf')
    pars = mod.guess(y0, x=x0)
    pars['amplitude'].set(value=0.8, min=0.6, max=0.83)
    pars['center1'].set(value=7, min=-23, max=20)
    pars['center2'].set(value=46, min=20, max=60)
    pars['sigma1'].set(value=7, min=1, max=30)
    pars['sigma2'].set(value=5, min=1, max=30)
    out = mod.fit(y, pars, x=x)
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
    # get r2 score
    correlation_matrix = np.corrcoef(y, Ayy)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    # Here the residual of the dataset is calculated
    resid = (y - Ayy)
    # Generate the mean absolute error
    MAE = (np.sum(abs(resid))/len(resid))
    # Generate the RMSE
    test2 = np.zeros(len(resid))
    for i in range(0,len(resid)):
        test2[i] = float(resid.iloc[i])**2
    RMSE = (np.sum(test2)/len(resid))**0.5
    # Generate Willmont index
    if (np.sum(abs(Ayy - y))) > 2*np.sum(abs(y - np.mean(y))):
        Wilm = ((2*np.sum(abs(y - np.mean(y))))/(np.sum(abs(Ayy - y)))) - 1
    else:
        Wilm = 1 - ((np.sum(abs(Ayy - y)))/(2*np.sum(abs(y - np.mean(y)))))
    return (r_squared, MAE, RMSE, Wilm, A, m1, s1, m2, s2, np.min(dataset['HeatMid']), np.max(dataset['HeatMid']))
# %%
pft_data_span = np.zeros((9,16))
for i in range(1, 17):
    dataset = PSIIContr[PSIIContr['Adjusted PFT'] == i]
    if dataset.shape[0] > 25:
        pft_data_span[:,i-1] = carlo_stats(dataset)[:9]

bad_stats = [[2,7,10,11]]

for i in range(0,9):
    print(np.mean(np.delete(pft_data_span[i,:], bad_stats)))
# %%
#####################################################################

# Tol - Res Trade-Off Plot (See below)

#####################################################################
par_spsht = pd.read_excel('param-spreadsheet.xlsx', engine='openpyxl')
par_spsht['PFT'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,99,99,99,99]

adj_par = par_spsht.drop([2, 7, 10, 11, 16, 17, 18, 19])
adj_par['T_range'] = adj_par['T_MH'] - adj_par['T_MC']

sort_par = adj_par.sort_values(by=['T_range'], ascending=False, ignore_index=True)
heights = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12]
# %%
def rect(x, x1, x2):
    return 1 if (x1 < x) & (x < x2) else 0

x = np.arange(-5, 45, 0.5)
yy = np.zeros((12, 100))
for j in range(1,12):
    for i in range(0, 100):
        yy[j,i] = rect(x[i], sort_par['T_MC'].iloc[12 - j], sort_par['T_MH'].iloc[12 - j])

y_stack = np.cumsum(yy, axis=0)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.fill_between(x, 0, y_stack[0,:], alpha=.7)
ax1.fill_between(x, y_stack[0,:], y_stack[1,:], alpha=.7)
ax1.fill_between(x, y_stack[1,:], y_stack[2,:])
ax1.fill_between(x, y_stack[2,:], y_stack[3,:])
ax1.fill_between(x, y_stack[3,:], y_stack[4,:])
ax1.fill_between(x, y_stack[4,:], y_stack[5,:])
ax1.fill_between(x, y_stack[5,:], y_stack[6,:])
ax1.fill_between(x, y_stack[6,:], y_stack[7,:])
ax1.fill_between(x, y_stack[7,:], y_stack[8,:])
ax1.fill_between(x, y_stack[8,:], y_stack[9,:])
ax1.fill_between(x, y_stack[9,:], y_stack[10,:])
ax1.fill_between(x, y_stack[10,:], y_stack[11,:])

plt.show()
# %%
x = np.arange(-5, 45, 0.5)
m_T_MC = np.mean(sort_par['T_MC'])
m_T_MH = np.mean(sort_par['T_MH'])
for i in range(0,12):
    plt.hlines(heights[i], sort_par['T_MC'].iloc[i], sort_par['T_MH'].iloc[i], label=sort_par['PFT'].iloc[i])
plt.ylim([-15, 10])
plt.xlim([-5, 55])
plt.vlines([m_T_MC, m_T_MH], -15, 10, color='grey')
plt.vlines([m_T_MC - np.std(sort_par['T_MC'])], -15, 0, linestyle='dashed', color='cyan')
plt.vlines([m_T_MH + np.std(sort_par['T_MH'])], -15, 0, linestyle='dashed', color='orange')
plt.legend()
plt.fill_between(x, 0, y_stack[11,:])
plt.title('Tolerance ranges of PFTs')
plt.xlabel('Temperature [C]')
#ax = plt.gca()
#ax.axes.yaxis.set_ticklabels([])

# %%
labels = sort_par['PFT'].iloc[::-1].tolist()
newlabel = [str(i) for i in labels]
men_means = sort_par['s1'].iloc[::-1].tolist()
women_means = sort_par['s2'].iloc[::-1].tolist()

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.barh(x - width/2, men_means, width, label='s1')
rects2 = ax.barh(x + width/2, women_means, width, label='s2')

ax.set_xlabel('Resiliency parameter')
ax.set_ylabel('PFT #')
ax.set_title('PFT resiliency response')
ax.set_yticks(x)
ax.set_yticklabels(newlabel)
ax.legend()

plt.show()
# %%
# 3-10-22 Starting work on the T_i index
# Work with Ordered_set , which has LTval, HTval, CTval included
# These are the results of the Climate ANOVA in terms of ratios (See excel on GDoc)
# Should do this within the code, but would involve more complex
# use of the ANOVA function
from scipy.optimize import curve_fit

# Create residual values for fitting
Ordered_set['resid'] = Ordered_set['phiPSIImax']
for i in range(0, len(Ordered_set['phiPSIImax'])):
    pt = Ordered_set['PFT #'].iloc[i]
    if pt == 71:
        Ordered_set['resid'].iloc[i] = np.nan 
    else:
        a, c1, c2, s1, s2 = par_spsht.iloc[pt - 1, 1:6]
        x = Ordered_set['HeatMid'].iloc[i]
        y = Ordered_set['phiPSIImax'].iloc[i]
        Ordered_set['resid'].iloc[i] = point_resid(x, y, a, c1, s1, c2, s2)

# Order: LT, HT, CT, LxH, LxC, HxC, LHC
LT_ = 8.78 / 100
HT_ = 1.28 / 100
CT_ = 6.42 / 100
LxH_ = 10.43 / 100
LxC_ = 20.43 / 100
HxC_ = 9.76 / 100
LHC_ = 36.78 / 100

# 3-15 Try a mean value offset for each index, i.e. lt[i] - mean(lt[:])

def T_index(lt, ht, ct, a1, a2, a3, a4, a5, a6, a7):
    return a1*lt + a2*ht + a3*ct + a4*lt*ht + a5*lt*ct + a6*ht*ct + a7*lt*ht*ct

i1m = np.mean(Ordered_set['LTval'])
i2m = np.mean(Ordered_set['HTval'])
i3m = np.mean(Ordered_set['CLval'])
tryalstep1 = T_index(Ordered_set['LTval'] - i1m, Ordered_set['HTval'] - i2m, Ordered_set['CLval'] - i3m, LT_,HT_,CT_,LxH_,LxC_,HxC_,LHC_)
Ordered_set['T_i'] = tryalstep1

# %%
# Global set
multi_index = pd.read_csv('multi_index.csv') #w,wmean,wcold,p,pmean,pheat,clim
crrtmulti = np.reshape(np.array(multi_index.iloc[:,1:]), (360,720,8))

lme = np.mean(crrtmulti[:,:,1])
hme = np.mean(crrtmulti[:,:,4])
clm = np.mean(crrtmulti[:,:,6])
glob_Ti = T_index(crrtmulti[:,:,1] -lme, crrtmulti[:,:,4]-hme, crrtmulti[:,:,6]-clm,LT_,HT_,CT_,LxH_,LxC_,HxC_,LHC_)
#
ad_glob_Ti = T_index(crrtmulti[:,:,1] -i1m, crrtmulti[:,:,4]-i2m, crrtmulti[:,:,6]-i3m,LT_,HT_,CT_,LxH_,LxC_,HxC_,LHC_)

# %%
def Ti_method(dataset, percent):
    gap = int(100 - percent)

    ltmean_set =  pd.DataFrame(np.zeros((13,gap)))

    for i in range(0, gap):
        a = i
        b = percent + i

        low_q1 = np.quantile(Ordered_set['T_i'][:], a/100)
        high_q1 = np.quantile(Ordered_set['T_i'][:], b/100)

        new_set1 = dataset[(dataset['T_i'] > low_q1) & (dataset['T_i'] < high_q1)]

        n1 = len(new_set1['HeatMid'])

        ltmean_set.iloc[0,i] = (a + b)/2
        ltmean_set.iloc[1,i] = n1
        ltmean_set.iloc[2,i] = np.min(new_set1['HeatMid'])
        ltmean_set.iloc[3,i] = np.max(new_set1['HeatMid'])
        if n1 > 30:
            ltmean_set.iloc[4:9,i] = get_set_resid(new_set1)[2]['val'][:]
            ltmean_set.iloc[9:,i] = removed_PFT_resid(new_set1)[3,:]
        else:
            ltmean_set.iloc[4:,i] = -999     

    return ltmean_set

# %%
#########################
# Can Skip
#########################
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8,10))
for q in [75, 66, 50, 33, 25]:
    try31 = Ti_method(Ordered_set, q)

    a = ax[0,0].scatter(try31.iloc[5,:] + 2*try31.iloc[6,:], try31.iloc[0,:], c=try31.iloc[2,:], cmap='jet')

    b = ax[0,1].scatter(try31.iloc[7,:] - 2*try31.iloc[8,:], try31.iloc[0,:], c=try31.iloc[3,:], cmap='jet')
    ax[0,0].set_xlim((-25, 35))
    ax[0,1].set_xlim((20, 57))
    
    c = ax[1,0].scatter(try31.iloc[5,:], try31.iloc[0,:], c=try31.iloc[2,:], cmap='jet')
    
    d = ax[1,1].scatter(try31.iloc[7,:], try31.iloc[0,:], c=try31.iloc[3,:], cmap='jet')
    ax[1,0].set_xlim((-25, 35))
    ax[1,1].set_xlim((20, 57))
    
    e = ax[2,0].scatter(try31.iloc[6,:], try31.iloc[0,:], c=try31.iloc[2,:], cmap='jet')
    
    f = ax[2,1].scatter(try31.iloc[8,:], try31.iloc[0,:], c=try31.iloc[3,:], cmap='jet')
    ax[2,0].set_xlim((0, 30))
    ax[2,1].set_xlim((0, 30))
    
plt.colorbar(a, ax=ax[0,0])
#ax[0,0].set_yticklabels((-41.8,-10.4,4.6,16,21.4))
plt.colorbar(b, ax=ax[0,1])
#ax[0,1].set_yticklabels((-41.8,-10.4,4.6,16,21.4))
plt.colorbar(c, ax=ax[1,0])
#ax[1,0].set_yticklabels((6.2, 11.1, 17.4, 23.3, 27.1))
plt.colorbar(d, ax=ax[1,1])
#ax[1,1].set_yticklabels((6.2, 11.1, 17.4, 23.3, 27.1))
plt.colorbar(e, ax=ax[2,0])
#ax[2,0].set_yticklabels((-14.5, 2.4, 10.1, 20.2, 24.2))
plt.colorbar(f, ax=ax[2,1])
#ax[2,1].set_yticklabels((-14.5, 2.4, 10.1, 20.2, 24.2))
# %%
############################################
# Need to clarify and make sure my r2 method for the fitting lines is correct
ti_frame = Ti_method(Ordered_set, 75)
for q in [66, 50, 33, 25]:
    try31 = Ti_method(Ordered_set, q)
    ti_frame = pd.concat([ti_frame, try31], axis=1)

ti_T = ti_frame.T
ti_T['13'] = np.quantile(Ordered_set['T_i'][:], ti_frame.iloc[0,:]/100)
# %%
def linef(a,b,x):
    return a + b*x
# %%
x = np.arange(-1000, 1000, 0.5)
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8,10), sharex=True, tight_layout=True)
a = ax[0,0].scatter(ti_T.iloc[:,13], ti_T.iloc[:,5] + 2*ti_T.iloc[:,6], c=ti_T.iloc[:,2], cmap='jet')
#ax[0,0].plot(x, linef(21.9555, -0.00176, x), '-r', label='r2 = 0.03')
ax[0,0].plot(x, linef(22.345, -0.00386, x), '-b', label='y = 22.345 - 0.004 x\n' + r'$r^{2}$' + ' = 0.16')
ax[0,0].plot(x, linef(-3.08511, -0.003618, x) + 2*linef(12.7151, -0.00059, x), 'k', linestyle='dashed')
b = ax[0,1].scatter(ti_T.iloc[:,13], ti_T.iloc[:,7] - 2*ti_T.iloc[:,8], c=ti_T.iloc[:,3], cmap='jet')
#ax[0,1].plot(x, linef(36.607, 0.01035, x), '-r', label='r2 = 0.35')
ax[0,1].plot(x, linef(32.81778, 0.00973, x), '-r', label='y = 32.818 + 0.010 x\n' + r'$r^{2}$' + ' = 0.41')
ax[0,1].plot(x, linef(47.7533, 0.00208, x) - 2*linef(7.46777, -0.00383, x), 'k', linestyle='dashed')
ax[0,0].legend()
ax[0,1].legend()
ax[0,0].set_ylabel('Tolerance Limits')
ax[0,0].set_title('Cold Temperature Factors')
ax[0,1].set_title('Hot Temperature Factors')
c = ax[1,0].scatter(ti_T.iloc[:,13], ti_T.iloc[:,5], c=ti_T.iloc[:,2], cmap='jet')
#ax[1,0].plot(x, linef(-3.45388, -0.00389, x), '-r', label='r2 = 0.33')
#'y = 32.818 + 0.010 x\n' + r'$r^{2}$' + ' = 0.41'
ax[1,0].plot(x, linef(-3.08511, -0.003618, x), '-b', label='y = -3.085 - 0.004 x\n' + r'$r^{2}$' + ' = 0.49')
d = ax[1,1].scatter(ti_T.iloc[:,13], ti_T.iloc[:,7], c=ti_T.iloc[:,3], cmap='jet')
#ax[1,1].plot(x, linef(48.02599, 0.00301, x), '-r', label='r2 = 0.15')
ax[1,1].plot(x, linef(47.7533, 0.00208, x), '-r', label='y = 47.753 + 0.002 x\n' + r'$r^{2}$' + ' = 0.30')
ax[1,0].legend()
ax[1,1].legend()
ax[1,0].set_ylabel('T50 parameter ($m$)')
e = ax[2,0].scatter(ti_T.iloc[:,13], ti_T.iloc[:,6], c=ti_T.iloc[:,2], cmap='jet')
#ax[2,0].plot(x, linef(11.63704, 0.00439, x), '-r', label='r2 = 0.34')
ax[2,0].plot(x, linef(12.7151, -0.00059, x), '-b', label='y = 12.715 - 0.0006 x\n' + r'$r^{2}$' + ' = 0.001')
f = ax[2,1].scatter(ti_T.iloc[:,13], ti_T.iloc[:,8], c=ti_T.iloc[:,3], cmap='jet')
#ax[2,1].plot(x, linef(5.7095, -0.00367, x), '-r', label='r2 = 0.31')
ax[2,1].plot(x, linef(7.46777, -0.00383, x), '-r', label='y = 7.468 - 0.004 x\n' + r'$r^{2}$' + ' = 0.37')
ax[2,0].legend()
ax[2,1].legend()
ax[2,0].set_ylabel('Resiliency parameter ' + '($s$)')
fig.supxlabel('Climatology Temperature Index (CTI)')
fig.suptitle('Linear Modeling of Climatology Index on parameters')
plt.colorbar(a, ax=ax[0,0])
plt.colorbar(b, ax=ax[0,1])
plt.colorbar(c, ax=ax[1,0])
plt.colorbar(d, ax=ax[1,1])
plt.colorbar(e, ax=ax[2,0])
plt.colorbar(f, ax=ax[2,1])
# %%
########################################################

# FOR THESE SECTIONS, CONSIDER USING THE SCIPY STATS LINEAR REGRESSION
# TO VERIFY THE R2 RESULTS (seems to check out 6/9)

########################################################
# Making the linear regression method
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
regr = linear_model.LinearRegression() # LLS( y - (a - b*x) )
x = np.array(ti_T.iloc[:,5]) # [ti_T.iloc[:,7] < ti_T.iloc[:,3]]
y = np.array(ti_T.iloc[:,13]).reshape(-1,1)
model = regr.fit(y,x)

r_sq = model.score(y,x)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('linef(' + str(model.intercept_) + ',' + str(model.coef_) + ', x)')
x_pred = model.predict(y)
plt.scatter(y,x)
plt.plot(y, x_pred)
# %%
# Weighted linear regression method
regr = linear_model.LinearRegression()
y = np.array(ti_T.iloc[:,5])
x = np.array(ti_T.iloc[:,13]).reshape(-1,1)

w = (1 / (1 + abs(np.nanmin(ti_T.iloc[:,2]) - ti_T.iloc[:,2])))
model = regr.fit(x,y, w) # LLS (W (y - (a - b*x)))

r_sq = model.score(x,y, w)

print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('linef(' + str(model.intercept_) + ',' + str(model.coef_) + ', x)')
y_pred = model.predict(x)
plt.scatter(x,y, s=w*20, c=w)
plt.plot(x, y_pred)
# %%
x = np.arange(-1000, 1000, 0.5)
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,10), sharex=True, tight_layout=True)
a = ax[0,0].scatter(ti_T.iloc[:,13][ti_T.iloc[:,5] > ti_T.iloc[:,2]], ti_T.iloc[:,5][ti_T.iloc[:,5] > ti_T.iloc[:,2]] + 2*ti_T.iloc[:,6][ti_T.iloc[:,5] > ti_T.iloc[:,2]], c=ti_T.iloc[:,2][ti_T.iloc[:,5] > ti_T.iloc[:,2]], cmap='jet')
ax[0,0].plot(x, linef(22.029984139013802,[-0.00451319], x), '-g', label='y = 22.029 -0.0045 x\n' + r'$r^{2}$' + ' = 0.217')
ax[0,0].plot(x, linef(22.26720653232525,[-0.00445181], x), '-k', label='y = 22.267 -0.0044 x\n' + r'$r^{2}$' + ' = 0.199')
b = ax[0,1].scatter(ti_T.iloc[:,13][ti_T.iloc[:,7] < ti_T.iloc[:,3]], ti_T.iloc[:,7][ti_T.iloc[:,7] < ti_T.iloc[:,3]] - 2*ti_T.iloc[:,8][ti_T.iloc[:,7] < ti_T.iloc[:,3]], c=ti_T.iloc[:,3][ti_T.iloc[:,7] < ti_T.iloc[:,3]], cmap='jet')
ax[0,1].plot(x, linef(34.97416465002699,[0.00392954], x), '-g', label='y = 34.974 +0.0039 x\n' + r'$r^{2}$' + ' = 0.091')
ax[0,1].plot(x, linef(32.19239508838175,[0.00821771], x), '-k', label='y = 32.192 +0.0082 x\n' + r'$r^{2}$' + ' = 0.353')
ax[0,0].set_ylabel('Tolerance Limits')
ax[0,0].set_title('Cold Temperature Factors')
ax[0,1].set_title('Hot Temperature Factors')
c = ax[1,0].scatter(ti_T.iloc[:,13][ti_T.iloc[:,5] > ti_T.iloc[:,2]], ti_T.iloc[:,5][ti_T.iloc[:,5] > ti_T.iloc[:,2]], c=ti_T.iloc[:,2][ti_T.iloc[:,5] > ti_T.iloc[:,2]], cmap='jet')
ax[1,0].plot(x, linef(-3.0130224140820343,[-0.00269678], x), '-g', label='y = -3.013 -0.0026 x\n' + r'$r^{2}$' + ' = 0.375')
ax[1,0].plot(x, linef(-2.9921873933452683,[-0.00315343], x), '-k', label='y = -2.992 -0.0031 x\n' + r'$r^{2}$' + ' = 0.486')
d = ax[1,1].scatter(ti_T.iloc[:,13][ti_T.iloc[:,7] < ti_T.iloc[:,3]], ti_T.iloc[:,7][ti_T.iloc[:,7] < ti_T.iloc[:,3]], c=ti_T.iloc[:,3][ti_T.iloc[:,7] < ti_T.iloc[:,3]], cmap='jet')
ax[1,1].plot(x, linef(47.9322814762204,[0.00138557], x), '-g', label='y = 47.932 + 0.0013 x\n' + r'$r^{2}$' + ' = 0.452')
ax[1,1].plot(x, linef(47.65399716383878,[0.00175651], x), '-k', label='y = 47.653 + 0.0017 x\n' + r'$r^{2}$' + ' = 0.574')
ax[1,0].set_ylabel('T50 parameter ($m$)')
e = ax[2,0].scatter(ti_T.iloc[:,13][ti_T.iloc[:,5] > ti_T.iloc[:,2]], ti_T.iloc[:,6][ti_T.iloc[:,5] > ti_T.iloc[:,2]], c=ti_T.iloc[:,2][ti_T.iloc[:,5] > ti_T.iloc[:,2]], cmap='jet')
ax[2,0].plot(x, linef(12.521503276547914,[-0.0009082], x), '-g', label='y = 12.521 - 0.0009 x\n' + r'$r^{2}$' + ' = 0.032')
ax[2,0].plot(x, linef(12.629696962835256,[-0.00064919], x), '-k', label='y = 12.629 -0.0006 x\n' + r'$r^{2}$' + ' = 0.018')
f = ax[2,1].scatter(ti_T.iloc[:,13][ti_T.iloc[:,7] < ti_T.iloc[:,3]], ti_T.iloc[:,8][ti_T.iloc[:,7] < ti_T.iloc[:,3]], c=ti_T.iloc[:,3][ti_T.iloc[:,7] < ti_T.iloc[:,3]], cmap='jet')
ax[2,1].plot(x, linef(6.479058413096703,[-0.00127198], x), '-g', label='y = 6.479 -0.0012 x\n' + r'$r^{2}$' + ' = 0.048')
ax[2,1].plot(x, linef(7.730801037728514,[-0.0032306], x), '-k', label='y = 7.730 -0.0032 x\n' + r'$r^{2}$' + ' = 0.299')
ax[2,0].set_ylabel('Resiliency parameter ' + '($s$)')
fig.supxlabel('Climatology Temperature Index (CTI)')
fig.suptitle('Linear Modeling of Climatology Temperature Index on parameters')
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()
ax[2,0].legend()
ax[2,1].legend()
plt.colorbar(a, ax=ax[0,0])
plt.colorbar(b, ax=ax[0,1])
plt.colorbar(c, ax=ax[1,0])
plt.colorbar(d, ax=ax[1,1])
plt.colorbar(e, ax=ax[2,0])
plt.colorbar(f, ax=ax[2,1])
# %%
# Working on smoothing methods to improve r2

ti_Tn = ti_T.sort_values(by='13')
ti_Tn.reset_index()

width_chose = 300

trail1 = np.zeros((4,1000))
trail2 = np.zeros((4,1000))
trail3 = np.zeros((4,1000))
trail4 = np.zeros((1000))
for i in range(1000):
    trail1[0,i] = np.mean(ti_Tn.iloc[:,5][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail1[1,i] = np.mean(ti_Tn.iloc[:,6][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail1[2,i] = np.mean(ti_Tn.iloc[:,7][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail1[3,i] = np.mean(ti_Tn.iloc[:,8][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail2[0,i] = np.std(ti_Tn.iloc[:,5][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail2[1,i] = np.std(ti_Tn.iloc[:,6][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail2[2,i] = np.std(ti_Tn.iloc[:,7][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail2[3,i] = np.std(ti_Tn.iloc[:,8][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail3[0,i] = np.median(ti_Tn.iloc[:,5][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail3[1,i] = np.median(ti_Tn.iloc[:,6][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail3[2,i] = np.median(ti_Tn.iloc[:,7][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail3[3,i] = np.median(ti_Tn.iloc[:,8][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])
    trail4[i] = len(ti_Tn.iloc[:,5][(ti_Tn.iloc[:,13] > -1000 - width_chose + 2*i) & (ti_Tn.iloc[:,13] < -1000 + width_chose + 2*i)])


xs = np.linspace(-1000, 1000, 1000)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,10), sharex=True, tight_layout=True)
fig.suptitle('SMOOTHING (chosen width = 2*' + str(width_chose) + ')')

ax[0,0].set_title('param 5 (m1)')
ax[0,1].set_title('param 6 (s1)')
ax[1,0].set_title('param 7 (m2)')
ax[1,1].set_title('param 8 (s2)')

ax[0,0].plot(xs, trail1[0,:], 'r', label='mean')
ax[0,1].plot(xs, trail1[1,:], 'r')
ax[1,0].plot(xs, trail1[2,:], 'r')
ax[1,1].plot(xs, trail1[3,:], 'r')

ax[0,0].plot(xs, trail3[0,:], 'k', linestyle='dashed', label='median')
ax[0,1].plot(xs, trail3[1,:], 'k', linestyle='dashed')
ax[1,0].plot(xs, trail3[2,:], 'k', linestyle='dashed')
ax[1,1].plot(xs, trail3[3,:], 'k', linestyle='dashed')

ax[0,0].fill_between(xs, trail1[0,:] - trail2[0,:], trail1[0,:] + trail2[0,:], alpha=0.4, label='std')
ax[0,1].fill_between(xs, trail1[1,:] - trail2[1,:], trail1[1,:] + trail2[1,:], alpha=0.4)
ax[1,0].fill_between(xs, trail1[2,:] - trail2[2,:], trail1[2,:] + trail2[2,:], alpha=0.4)
ax[1,1].fill_between(xs, trail1[3,:] - trail2[3,:], trail1[3,:] + trail2[3,:], alpha=0.4)

ax[0,0].plot(xs, linef(-4.314, -0.00265, xs), 'g', label='r2 = 0.396')
ax[0,1].plot(xs, linef(12.494, 0.00226, xs), 'g', label='r2 = 0.456')
ax[1,0].plot(xs, linef(47.923, 0.00226, xs), 'g', label='r2 = 0.634')
ax[1,1].plot(xs, linef(5.812, -0.0045, xs), 'g', label='r2 = 0.843')

ax[0,0].scatter(ti_Tn.iloc[:,13], ti_Tn.iloc[:,5], c='b')
ax[0,1].scatter(ti_Tn.iloc[:,13], ti_Tn.iloc[:,6], c='b')
ax[1,0].scatter(ti_Tn.iloc[:,13], ti_Tn.iloc[:,7], c='b')
ax[1,1].scatter(ti_Tn.iloc[:,13], ti_Tn.iloc[:,8], c='b')

ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()

# %%
# Demonstrates the unequal number of points, suggesting a different progression method may be preferred
plt.title('Number of QSA points in +/- width range')
plt.plot(xs, trail4[:])

# %%
regr = linear_model.LinearRegression() # LLS( y - (a - b*x) )
x = np.array(trail3[3,:]) # m2 - 2* s2
y = np.array(xs).reshape(-1,1)
model = regr.fit(y,x)

r_sq = model.score(y,x)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('linef(' + str(model.intercept_) + ',' + str(model.coef_) + ', x)')
x_pred = model.predict(y)
# %%
def point_width_smoothing(dataset, width, method=0):
    '''ensure that dataset is formed from ti_T based column structure'''
    point_set_for_curve = np.zeros((4,len(dataset.iloc[:,0]))) # gen 0 for each point assigning
    if method == 1:
        for i in range(len(dataset.iloc[:,0])):
            point_set_for_curve[0,i] = np.nanmedian(dataset.iloc[:,5][(dataset.iloc[:,13] < dataset.iloc[i,13] + width) & (dataset.iloc[:,13] > dataset.iloc[i,13] - width)])
            point_set_for_curve[1,i] = np.nanmedian(dataset.iloc[:,6][(dataset.iloc[:,13] < dataset.iloc[i,13] + width) & (dataset.iloc[:,13] > dataset.iloc[i,13] - width)])
            point_set_for_curve[2,i] = np.nanmedian(dataset.iloc[:,7][(dataset.iloc[:,13] < dataset.iloc[i,13] + width) & (dataset.iloc[:,13] > dataset.iloc[i,13] - width)])
            point_set_for_curve[3,i] = np.nanmedian(dataset.iloc[:,8][(dataset.iloc[:,13] < dataset.iloc[i,13] + width) & (dataset.iloc[:,13] > dataset.iloc[i,13] - width)])
    else:
        for i in range(len(dataset.iloc[:,0])):
            point_set_for_curve[0,i] = np.nanmean(dataset.iloc[:,5][(dataset.iloc[:,13] < dataset.iloc[i,13] + width) & (dataset.iloc[:,13] > dataset.iloc[i,13] - width)])
            point_set_for_curve[1,i] = np.nanmean(dataset.iloc[:,6][(dataset.iloc[:,13] < dataset.iloc[i,13] + width) & (dataset.iloc[:,13] > dataset.iloc[i,13] - width)])
            point_set_for_curve[2,i] = np.nanmean(dataset.iloc[:,7][(dataset.iloc[:,13] < dataset.iloc[i,13] + width) & (dataset.iloc[:,13] > dataset.iloc[i,13] - width)])
            point_set_for_curve[3,i] = np.nanmean(dataset.iloc[:,8][(dataset.iloc[:,13] < dataset.iloc[i,13] + width) & (dataset.iloc[:,13] > dataset.iloc[i,13] - width)])

    return point_set_for_curve

def curve_set_get_r2(dataset, CTI_values):
    '''dataset is result from point_width_smoothing
    CTI_values are the CTI values from the previous points that 
    have now been 'smoothed'. '''
    regr = linear_model.LinearRegression()
    # create a array to save the values to (inter, slope, r2) for 4 params
    curve_results = np.zeros((4,3)) # 4 params (i) 3 values (j) [i,j]
    y = np.array(CTI_values).reshape(-1,1) # same for all
    for i in range(4):
        x = np.array(dataset[i,:]) # m2 - 2* s2
        model = regr.fit(y,x)
        r_sq = model.score(y,x)

        curve_results[i,:] = [r_sq, model.intercept_, model.coef_]
    
    return curve_results, np.std(dataset, axis=1)

# %%
widths = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500]
rsquares_over_widths = np.zeros((len(widths), 4, 3))
sigmas_by_width = np.zeros((len(widths), 4))
for count, value in enumerate(widths):
    rsquares_over_widths[count,:,:], sigmas_by_width[count,:] = curve_set_get_r2(point_width_smoothing(ti_T, value), ti_T.iloc[:,13])[:]
# %%
fig = plt.figure(figsize=(12,12), constrained_layout=True)

gs = fig.add_gridspec(3,2)
ax1 = fig.add_subplot(gs[0, 0])
ax01 = ax1.twinx()
ax2 = fig.add_subplot(gs[0, 1])
ax02 = ax2.twinx()
ax3 = fig.add_subplot(gs[1, 0])
ax03 = ax3.twinx()
ax4 = fig.add_subplot(gs[1, 1])
ax04 = ax4.twinx()
ax5 = fig.add_subplot(gs[2, :])

ax1.set_title('r2 sens. of m1 (param5)')
ax2.set_title('r2 sens. of s1 (param6)')
ax3.set_title('r2 sens. of m2 (param7)')
ax4.set_title('r2 sens. of s2 (param8)')
ax5.set_title('mean r2 for 4 params')

ax1.set_ylim([0,1])
ax2.set_ylim([0,1])
ax3.set_ylim([0,1])
ax4.set_ylim([0,1])
ax5.set_ylim([0,1])

ax1.set_ylabel('r-squared value')
ax2.set_ylabel('r-squared value')
ax3.set_ylabel('r-squared value')
ax4.set_ylabel('r-squared value')

ax01.set_ylabel('stand. devi.')
ax02.set_ylabel('stand. devi.')
ax03.set_ylabel('stand. devi.')
ax04.set_ylabel('stand. devi.')


ax1.plot(widths, rsquares_over_widths[:,0,0], label='rsquared')
ax2.plot(widths, rsquares_over_widths[:,1,0], label='rsquared')
ax3.plot(widths, rsquares_over_widths[:,2,0], label='rsquared')
ax4.plot(widths, rsquares_over_widths[:,3,0], label='rsquared')
ax01.plot(widths, sigmas_by_width[:,0], 'r', label='width std')
ax02.plot(widths, sigmas_by_width[:,1], 'r', label='width std')
ax03.plot(widths, sigmas_by_width[:,2], 'r', label='width std')
ax04.plot(widths, sigmas_by_width[:,3], 'r', label='width std')
#ax01.hlines(np.std(ti_T.iloc[:,5]), 0,500, label='raw std')
#ax02.hlines(np.std(ti_T.iloc[:,6]), 0,500, label='raw std')
#ax03.hlines(np.std(ti_T.iloc[:,7]), 0,500, label='raw std')
#ax04.hlines(np.std(ti_T.iloc[:,8]), 0,500, label='raw std')
ax5.plot(widths, np.mean(rsquares_over_widths[:,:,0], axis=1))
# %%
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10), tight_layout=True)
ax[0,0].scatter(rsquares_over_widths[:,0,1], rsquares_over_widths[:,0,2], s=(10*rsquares_over_widths[:,0,0])**2, c=widths, cmap='seismic')
ax[0,1].scatter(rsquares_over_widths[:,1,1], rsquares_over_widths[:,1,2], s=(10*rsquares_over_widths[:,1,0])**2, c=widths, cmap='seismic')
ax[1,0].scatter(rsquares_over_widths[:,2,1], rsquares_over_widths[:,2,2], s=(10*rsquares_over_widths[:,2,0])**2, c=widths, cmap='seismic')
ax[1,1].scatter(rsquares_over_widths[:,3,1], rsquares_over_widths[:,3,2], s=(10*rsquares_over_widths[:,3,0])**2, c=widths, cmap='seismic')

ax[0,0].plot(rsquares_over_widths[:,0,1], rsquares_over_widths[:,0,2])
ax[0,1].plot(rsquares_over_widths[:,1,1], rsquares_over_widths[:,1,2])
ax[1,0].plot(rsquares_over_widths[:,2,1], rsquares_over_widths[:,2,2])
ax[1,1].plot(rsquares_over_widths[:,3,1], rsquares_over_widths[:,3,2])
# %%
def assess_function_adj(dataset, width):
    '''ensure that dataset is formed from ti_T based column structure'''
    mind = np.nanmin(dataset.iloc[:,13])
    maxd = np.nanmax(dataset.iloc[:,13])
    point_set_for_curve = np.zeros((len(dataset.iloc[:,0]), 4, 2)) # gen 0 for each point assigning
    for i in range(len(dataset.iloc[:,0])):
        means = np.array()
        stds = np.array()
        for j in range(0,4):
            point_set_for_curve[j,i,0] = np.nanmean(dataset.iloc[:,5 + j][(dataset.iloc[:,13] < dataset.iloc[i,13] + width) & (dataset.iloc[:,13] > dataset.iloc[i,13] - width)])
            point_set_for_curve[j,i,1] = np.nanstd(dataset.iloc[:,5 + j][(dataset.iloc[:,13] < dataset.iloc[i,13] + width) & (dataset.iloc[:,13] > dataset.iloc[i,13] - width)])
        
            center1 = dataset.iloc[i,13]
            dist_to_ = [(center1 - width) - mind]
            num_bins_dir = dist_to_ / width
            edge = center1 - (math.ceil(num_bins_dir) * 2 * width + 1)

            while edge < maxd:
                means = means.append(np.nanmean(dataset.iloc[:,5+j][(dataset.iloc[:,13] < edge + 2*width) & (dataset.iloc[:,13] > edge)]))
                stds = stds.append(np.nanstd(dataset.iloc[:,5+j][(dataset.iloc[:,13] < edge + 2*width) & (dataset.iloc[:,13] > edge)]))
            



    return point_set_for_curve

def curve_set_get_r2_adj(dataset, CTI_values):
    '''dataset is result from point_width_smoothing
    CTI_values are the CTI values from the previous points that 
    have now been 'smoothed'. '''
    regr = linear_model.LinearRegression()
    # create a array to save the values to (inter, slope, r2) for 4 params
    curve_results = np.zeros((4,3)) # 4 params (i) 3 values (j) [i,j]
    y = np.array(CTI_values).reshape(-1,1) # same for all
    for i in range(4):
        x = np.array(dataset[i,:]) # m2 - 2* s2
        model = regr.fit(y,x)
        r_sq = model.score(y,x)

        curve_results[i,:] = [r_sq, model.intercept_, model.coef_]
    
    return curve_results, np.std(dataset, axis=1)

# %%
# potential attempt at creating loess method?
# https://rafalab.github.io/dsbook/smoothing.html








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

def Ti_offset_resid(dataset):
    '''Needs to include a Ti column value'''
    Ordered = dataset.sort_values(by='HeatMid')
    x = Ordered['HeatMid']
    x0 = x.iloc[:]
    y = Ordered['phiPSIImax']
    y0 = y.iloc[:]
    mod = RectangleModel(form='erf')
    pars = mod.guess(y0, x=x0)
    pars['amplitude'].set(value=0.8, min=0.6, max=0.83)
    pars['center1'].set(value=-3, min=-23, max=15)
    pars['center2'].set(value=46, min=35, max=57)
    pars['sigma1'].set(value=7, min=1, max=25)
    pars['sigma2'].set(value=5, min=1, max=12)
    out = mod.fit(y, pars, x=x)
    ps = get_Mod_paramsValues(out)
    A, m1, s1, m2, s2 = ps.val[0], ps.val[1], ps.val[2], ps.val[3], ps.val[4] 
    mid_t = ((m1 + 2*s1) + (m2 - 2*s2))/2
    # Current front is the original weighted linear regression method
    m1 = linef((-3.08511), (-0.003618), Ordered['T_i'])#linef(-4.314, -0.00265, x)#linef(-2.9921873933452683,[-0.00315343], x)#
    s1 = linef((11.63704), (0.00439), Ordered['T_i'])#linef(12.494, 0.00226, x)#linef(12.521503276547914,[-0.0009082], x)#
    m2 = linef((47.7533), (0.00208), Ordered['T_i'])#linef(47.923, 0.00226, x)#linef(47.9322814762204,[0.00138557], x)#
    s2 = linef((7.46777), (-0.00383), Ordered['T_i'])#linef(5.812, -0.0045, x)#linef(7.730801037728514,[-0.0032306], x)#
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

    m1p = m1 / (-3.08511)
    s1p = s1 / (11.63704)
    m2p = m2 / (47.7533)
    s2p = s2 / (7.46777)

    Ay2 = []
    for i in range(len(dataset['HeatMid'])):
        pft = int(dataset['Adjusted PFT'].iloc[i])
        Ay2.append(point_resid(x0.iloc[i], y0.iloc[i],
                    param_span[0,pft-1], param_span[1,pft-1]*m1p.iloc[i], param_span[2,pft-1]*s1p.iloc[i],
                    param_span[3,pft-1]*m2p.iloc[i], param_span[4,pft-1]*s2p.iloc[i]))

    
    Aa1p = (x - m1p)/s1p
    Aaa1p = []
    for i in range(len(Aa1p)):
            Aaa1p.append(math.erf(Aa1p.iloc[i]))
    Aa2p = -(x - m2p)/s2p
    Aaa2p = []
    for i in range(len(Aa2p)):
            Aaa2p.append(math.erf(Aa2p.iloc[i]))
    Ay3 = []
    for i in range(len(Aaa1p)):
            Ay3.append((A/2)* (Aaa1p[i] + Aaa2p[i]))

    return y - Ayy, Ayy, Ay2, Ay3
# %%
# Need to fix
#a = plt.scatter(Ordered_set['HeatMid'], Ti_offset_resid(Ordered_set)[1], c=Ordered_set['T_i'])
a = plt.scatter(Ti_offset_resid(Ordered_set)[1], Ti_offset_resid(Ordered_set)[2], c=Ordered_set['T_i'])
#plt.scatter(Ordered_set['HeatMid'], Ordered_set['phiPSIImax'])
plt.xlabel('Temperature [C]')
plt.ylabel('Fv/Fm')
plt.colorbar(a, label='T_i')
# %%
fig, axs = plt.subplots(2,2, tight_layout=True, figsize=(9,7))
axs[0,0].set_title('m1 spread')
axs[0,0].hist(linef(-3.45388, -0.00389, Ordered_set['T_i']))
axs[0,0].vlines(-3.55,0,300, 'g')
axs[0,1].set_title('m2 spread')
axs[0,1].hist(linef(48.02599, 0.00301, Ordered_set['T_i']))
axs[0,1].vlines(47.97, 0,300,'g')
axs[1,0].set_title('s1 spread')
axs[1,0].hist(linef(12.70469, 0.00106, Ordered_set['T_i']))
axs[1,0].vlines(11.7, 0,300,'g')
axs[1,1].set_title('s2 spread')
axs[1,1].hist(linef(5.7095, -0.00367, Ordered_set['T_i']))
axs[1,1].vlines(7.87,0,300,'g')
fig.suptitle('Distribution of parameters when defined by T_i', fontsize=16)
# %%
plt.hist(linef(-3.45388, -0.00389, Ordered_set['T_i']))
plt.vlines(-3.55,0,300, 'g')

# %%
# this shows how the distribution of the residuals differs between methods
fig, axs = plt.subplots(1,2, tight_layout=True, sharey=True, sharex=True)
axs[0].hist(Ti_offset_resid(Ordered_set[abs(Ordered_set['T_i']) < 1000])[0])
axs[0].set_title('Ti offset')
axs[1].hist(removed_PFT_resid(Ordered_set[abs(Ordered_set['T_i']) < 1000])[:,3])
axs[1].set_title('PFT offset')
fig.supxlabel('Residual')
fig.suptitle('Comparison of | Ti | < 1100 residual distribution')
# %%
sep_array_resid = [
    abs(removed_PFT_resid(Ordered_set[(abs(Ordered_set['T_i']) < 1000) & (Ordered_set['HeatMid'] < 15)])[:,3]) - abs(Ti_offset_resid(Ordered_set[(abs(Ordered_set['T_i']) < 1000) & (Ordered_set['HeatMid'] < 15)])[0]),
    abs(removed_PFT_resid(Ordered_set[(abs(Ordered_set['T_i']) < 1000) & (Ordered_set['HeatMid'] > 15) & (Ordered_set['HeatMid'] < 30)])[:,3]) - abs(Ti_offset_resid(Ordered_set[(abs(Ordered_set['T_i']) < 1000) & (Ordered_set['HeatMid'] > 15) & (Ordered_set['HeatMid'] < 30)])[0]),
    abs(removed_PFT_resid(Ordered_set[(abs(Ordered_set['T_i']) < 1000) & (Ordered_set['HeatMid'] > 30)])[:,3]) - abs(Ti_offset_resid(Ordered_set[(abs(Ordered_set['T_i']) < 1000) & (Ordered_set['HeatMid'] > 30)])[0])
]
colors = ['< 15', '15 < T < 30', '> 30']
plt.hist(sep_array_resid, 25, histtype='bar', density=True, label=colors)

mu = [np.mean(sep_array_resid[0]), np.mean(sep_array_resid[1]), np.mean(sep_array_resid[2])]
sigma = [np.std(sep_array_resid[0]), np.std(sep_array_resid[1]), np.std(sep_array_resid[2])]
x = np.linspace(-0.4,0.6,200)
plt.plot(x, 1/(sigma[0] * np.sqrt(2 * np.pi)) *
               np.exp( - (x - mu[0])**2 / (2 * sigma[0]**2) ), color='b', label='<15')
plt.plot(x, 1/(sigma[1] * np.sqrt(2 * np.pi)) *
               np.exp( - (x - mu[2])**2 / (2 * sigma[1]**2) ), color='orange', label='15<T<30')
plt.plot(x, 1/(sigma[2] * np.sqrt(2 * np.pi)) *
               np.exp( - (x - mu[2])**2 / (2 * sigma[2]**2) ), color='g', label='>30')

plt.legend()
plt.title('| PFT resid | - | CTI resid | (weighted original)')
plt.show()
# %%
# Deminstrating how adjusting the Ti cut off changes the residual between the 2 methods (PFT v Ti)
##############################
# Figure 5
##############################
x = np.arange(0,1600,50)
huh = []
huh2 = []
huh3 = []
for i in range(4,len(x)):
    huh.append(np.sum(np.abs(Ti_offset_resid(Ordered_set[abs(Ordered_set['T_i']) < x[i]])[0])))
    huh2.append(np.sum(np.abs(removed_PFT_resid(Ordered_set[abs(Ordered_set['T_i']) < x[i]])[:,3])))
    huh3.append(np.sum(np.abs(Ti_offset_resid(Ordered_set[abs(Ordered_set['T_i']) < x[i]])[0])) - np.sum(np.abs(removed_PFT_resid(Ordered_set[abs(Ordered_set['T_i']) < x[i]])[:,3])))

plt.plot(x[4:],huh, label='Ti-based residual sum')
plt.plot(x[4:],huh2, label='PFT-based residual sum')
plt.xlim([100,1500])
plt.legend()
plt.title('Comparison of sum of residuals')
plt.xlabel('Ti cutoff value')
plt.ylabel('Sum of residuals')
plt.grid()
#plt.plot(x[4:], huh3)
#plt.grid(True)
# %%
# Plot of histogram of residuals
hist, bins = np.histogram(Ti_offset_resid(Ordered_set[abs(Ordered_set['T_i']) < 1100])[2], bins=100, normed=True)
bin_centers = (bins[1:]+bins[:-1])*0.5
plt.plot(bin_centers, hist)
# %%
#####################################################################

# WMM Method, Dropped from study

#####################################################################

# Developing method for PFT-weight assembled model
# proof of concept
def mod_func(x, a, m1, s1, m2, s2):
    Aa1 = (x - m1)/s1
    Aaa1 = []
    for i in range(len(Aa1)):
            Aaa1.append(math.erf(Aa1[i]))
    Aa2 = -(x - m2)/s2
    Aaa2 = []
    for i in range(len(Aa2)):
            Aaa2.append(math.erf(Aa2[i]))
    Ayy = []
    for i in range(len(Aaa1)):
            Ayy.append((a/2)* (Aaa1[i] + Aaa2[i]))
    return Ayy

def mod_funcDF(x, a, m1, s1, m2, s2):
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
            Ayy.append((a/2)* (Aaa1[i] + Aaa2[i]))
    return Ayy
# %%
x = np.arange(-20, 60, 1)
a = mod_func(x, 0.8, -9, 6, 50, 4)
b = mod_func(x, 1, 0, 6, 50, 4)
c = np.zeros(len(a))
for i in range(len(a)):
    c[i] = (a[i] + b[i])/2
d = mod_func(x, (0.8 + 1)/2, (0 - 9)/2, (0 + 2*6 - (0 - 9)/2)/2, 50, 4)
plt.plot(x, a)
plt.plot(x, b)
plt.plot(x, c)
plt.plot(x, d)
# %%
#param_span[0,pft-1]

e = mod_func(x, param_span[0,6], param_span[1,6], param_span[2,6], param_span[3,6], param_span[4,6])
f = mod_func(x, param_span[0,9], param_span[1,9], param_span[2,9], param_span[3,9], param_span[4,9])
g = mod_func(x, param_span[0,14], param_span[1,14], param_span[2,14], param_span[3,14], param_span[4,14])
h = np.zeros(len(x))
for i in range(len(h)):
    h[i] = (e[i] + 3*f[i] + 7*g[i])/11
plt.plot(x,e, label='PFT 5, w=1')
plt.plot(x,f, label='PFT 8, w=3')
plt.plot(x,g, label='PFT 13, w=7')
#plt.plot(x, h)
plt.legend()
# %%
j = mod_func(x, (param_span[0,6] + 3*param_span[0,9] + 7*param_span[0,14])/11,
             (param_span[1,6] + 3*param_span[1,9] + 7*param_span[1,14])/11,
             (np.max((param_span[1,6] + 2*param_span[2,6], 
                     param_span[1,9] + 2*param_span[2,9],
                     param_span[1,14] + 2*param_span[2,14])) - (param_span[1,6] + 3*param_span[1,9] + 7*param_span[1,14])/11)/2,
             (param_span[3,6] + 3*param_span[3,9] + 7*param_span[3,14])/11,
             (np.min((param_span[3,6] - 2*param_span[4,6], 
                     param_span[3,9] - 2*param_span[4,9],
                     param_span[3,14] - 2*param_span[4,14])) - (param_span[3,6] + 3*param_span[3,9] + 7*param_span[3,14])/11)/(-2))

plt.plot(x,h, label='raw averaging')
plt.plot(x,j, label='WMM method')
plt.legend()
###
plt.plot(x,h - j, label='(raw - WMM)')
plt.legend()
###
plt.plot(x,(h - j)/h, label='(raw - WMM)/raw')
plt.legend()

# %%
# Need to make a function that goes thru a dataset and finds the 'weights'
import random
#PSgroup1 = PSIIContr.iloc[random.sample(range(0,2102), 400),:]

def pft_num_counter(dataset):
    pie_split = np.zeros((16))
    for i in range(len(dataset['Adjusted PFT'])):
        pft = dataset['Adjusted PFT'].iloc[i]
        pie_split[pft - 1] = pie_split[pft - 1] + 1
    return pie_split

# %%
# Now try making the weighted mean model curve

def WMM_params(dataset):
    s_pft = pft_num_counter(dataset)
    bool_pft = np.zeros((16))
    for i in range(0,16):
        if s_pft[i] < 1:
            bool_pft[i] = np.nan
        else:
            bool_pft[i] = 1
    bool_pft[2] = np.nan
    bool_pft[7] = np.nan
    bool_pft[10] = np.nan
    bool_pft[11] = np.nan
    #print(bool_pft)
    amp = 0
    m1 = 0
    m2 = 0

    tmc = np.nanmax(bool_pft*(param_span[1,:] + 2*param_span[2,:]))
    #print(tmc)
    tmh = np.nanmin(bool_pft*(param_span[3,:] - 2*param_span[4,:]))
    #print(tmh)

    for i in range(0,16):
        amp = amp + s_pft[i]*param_span[0,i]
        m1 = m1 + s_pft[i]*param_span[1,i]
        m2 = m2 + s_pft[i]*param_span[3,i]

    ampf = amp / len(dataset['Adjusted PFT'])
    m1f = m1 / len(dataset['Adjusted PFT'])
    m2f = m2 / len(dataset['Adjusted PFT'])
    s1f = (tmc - m1f)/2
    s2f = (m2f - tmh)/2

    return ampf, m1f, s1f, m2f, s2f
# %%
PSgroup1 = PSIIContr.iloc[random.sample(range(0,2102), 200),:]

plt.scatter(PSgroup1['HeatMid'], PSgroup1['phiPSIImax'], alpha=0.3)
g = WMM_params(PSgroup1)
x = np.arange(-20, 60, 1)
mn = mod_func(x, g[0], g[1], g[2], g[3], g[4])
plt.plot(x,mn)
gn = get_set_resid(PSgroup1)[2]['val']
hm = mod_func(x, gn[0], gn[1], gn[2], gn[3], gn[4])
plt.plot(x, hm)

resid1 = PSgroup1['phiPSIImax'] - mod_funcDF(PSgroup1['HeatMid'], g[0], g[1], g[2], g[3], g[4])
resid2 = PSgroup1['phiPSIImax'] - mod_funcDF(PSgroup1['HeatMid'], gn[0], gn[1], gn[2], gn[3], gn[4])
print(np.mean(np.abs(resid1)), np.mean(np.abs(resid2)))
# %%
resid_diff = []
for i in range(300):
    PSgroup1 = PSIIContr.iloc[random.sample(range(0,2102), 100),:]
    g = WMM_params(PSgroup1)
    gn = get_set_resid(PSgroup1)[2]['val']
    resid1 = PSgroup1['phiPSIImax'] - mod_funcDF(PSgroup1['HeatMid'], g[0], g[1], g[2], g[3], g[4])
    resid2 = PSgroup1['phiPSIImax'] - mod_funcDF(PSgroup1['HeatMid'], gn[0], gn[1], gn[2], gn[3], gn[4])
    resid_diff = np.append(resid_diff, np.mean(np.abs(resid1)/PSgroup1['phiPSIImax']) - np.mean(np.abs(resid2)/PSgroup1['phiPSIImax'])) # can remove the mean(abs())
plt.hist(resid_diff)

# %%
#####################################################################

# Global Ti Analysis working with surface mask

#####################################################################

import xarray as xr
# CMIP_data = xr.load_dataset('c:/Users/pjneri/Desktop/surfdata_0.9x1.25_hist_16pfts_Irrig_CMIP6_simyr1850_c190214.nc')

PFT_thing = xr.load_dataset('c:/Users/pjneri/Desktop/mksrf_pft_0.5x0.5_simyr2005.c090313.nc')
pftcropmaybe = xr.load_dataset('c:/Users/pjneri/Desktop/mksrf_20pft_0.5x0.5_rc2000_simyr1990s.c110321.nc')
# %%
# SKIP THIS SECTION
######################

# cumulative trial to get the monthly peak and weak values for all gridpoints
Monthly = pd.read_csv('Glob_monthly.csv', chunksize=10)
# %%
df = pd.read_csv("Glob_monthly.csv", usecols = np.arange(4,8,1))
print(df)

# Convert to DataFrame
#P = pd.DataFrame(data=Precip)
# %%
# method for doing the p and w calculation locally
num_cols = 360 * 720 # make sure to start at '0'
p_w_global = np.zeros((num_cols, 2))
for i in range(0, 3*720):
    #print('low: ', i*120, ' high: ', 120 + i*120)
    if i%720 == 0:
        print(i)
    cols = np.arange(i*120, 120 + i*120, 1) + 1
    mean3 = np.zeros((12,120))
    df = pd.read_csv('Glob_monthly.csv', usecols = cols)
    mean3[0,:] = (np.mean((df.iloc[0,:], df.iloc[1,:], df.iloc[11,:])))
    for j in range(1,11):
        mean3[j,:] = (np.mean(df.iloc[j-1:j+1,:]))
    mean3[11,:] = (np.mean((df.iloc[10,:], df.iloc[11,:], df.iloc[0,:])))
    m3df = pd.DataFrame(data=mean3)
    for k in range(120):
        p_w_global[k + i*120,0] = m3df[m3df.iloc[:,k] == np.max(m3df.iloc[:,k])].index[0] 
        p_w_global[k + i*120,1] = m3df[m3df.iloc[:,k] == np.min(m3df.iloc[:,k])].index[0]
        
# %%
# Create a .csv to export back to HPC
P_W_Glob = pd.DataFrame(data = p_w_global)
P_W_Glob.to_csv('c:/Users/pjneri/Desktop/Glob_pw.csv')

# %%
cols = np.arange(0,5,1)
col = np.append(cols,12)
ggg = pd.read_csv('c:/Users/pjneri/Desktop/Glob_data.csv', usecols=cols)
# %%
minmaxv = pd.read_csv('minmaxvals.csv')
rearange = np.reshape(np.array(minmaxv.iloc[:,1:]), (360,720,24))
pw = pd.read_csv('c:/Users/pjneri/Desktop/Glob_pw.csv')

pwrearge = np.reshape(np.array(pw), (360,720,3))
pglob = pwrearge[:,:,1]
wglob = pwrearge[:,:,2]

pminmaxglobal = np.zeros((360,720,2))
wminmaxglobal = np.zeros((360,720,2))
for i in range(360):
    for j in range(720):
        p = int(pglob[i,j])
        w = int(wglob[i,j])
        wminmaxglobal[i,j,0] = np.min((rearange[i,j,2*(((w - 1) -1)%12)],
                                   rearange[i,j,2*(((w - 1) )%12)],
                                   rearange[i,j,2*(((w - 1) +1)%12)]))
        wminmaxglobal[i,j,1] = np.max((rearange[i,j,2*(((w - 1) -1)%12)+1],
                                   rearange[i,j,2*(((w - 1) )%12)+1],
                                   rearange[i,j,2*(((w - 1) +1)%12)+1]))
        pminmaxglobal[i,j,0] = np.min((rearange[i,j,2*(((p - 1) -1)%12)],
                                   rearange[i,j,2*(((p - 1) )%12)],
                                   rearange[i,j,2*(((p - 1) +1)%12)]))
        pminmaxglobal[i,j,1] = np.max((rearange[i,j,2*(((p - 1) -1)%12)+1],
                                   rearange[i,j,2*(((p - 1) )%12)+1],
                                   rearange[i,j,2*(((p - 1) +1)%12)+1]))
# %%
ghet = np.concatenate((pminmaxglobal, wminmaxglobal), axis=2)
pd.DataFrame(ghet.reshape((360,720*4))).to_csv('c:/Users/pjneri/Desktop/pmnmxwmnmxglob.csv')

# %%
# START FROM THIS SECTION
##############################################

multi_index = pd.read_csv('multi_index.csv') #w,wmean,wcold,p,pmean,pheat,clim
crrtmulti = np.reshape(np.array(multi_index.iloc[:,1:]), (360,720,8))

lme = np.mean(crrtmulti[:,:,1])
hme = np.mean(crrtmulti[:,:,4])
clm = np.mean(crrtmulti[:,:,6])
glob_Ti = T_index(crrtmulti[:,:,1] -lme, crrtmulti[:,:,4]-hme, crrtmulti[:,:,6]-clm,LT_,HT_,CT_,LxH_,LxC_,HxC_,LHC_)
adj_glob_Ti = np.roll(ad_glob_Ti, 360) # NOTE THE AD_GLOB_TI WHICH IS BASED ON LOCATION CLIM NOT GLOBAL CLIM
# Generate the TI variable in the surface mask data set
PFT_thing["TI_INDEX"] = (['lat', 'lon'], adj_glob_Ti)

lats = PFT_thing['LAT']
lons = PFT_thing['LON']

# Add a variable for max PFT coverage
PFT_thing["MAX_PFT"] = (['lat', 'lon'], np.argmax(np.array(PFT_thing['PCT_PFT'][1:,:,:]), axis=0))
# %%
# Better CTI plot
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
ax.set_global()
ax.set_title('Global Map of |CTI| values less then 4000', fontsize=18)
a = ax.contourf(lons,lats, PFT_thing['TI_INDEX'].where(abs(PFT_thing["TI_INDEX"]) < 3600), transform=ccrs.PlateCarree(central_longitude=0), cmap='jet', levels=18)
ax.contourf(lons,lats, PFT_thing['TI_INDEX'].where((590 > abs(PFT_thing['TI_INDEX'])) | (abs(PFT_thing['TI_INDEX']) < 610)), colors='black', hatches=['/'], levels=1,alpha=0.1, transform=ccrs.PlateCarree(central_longitude=0))
ax.coastlines()
plt.colorbar(a, fraction=0.023, pad=0.04)

# %%
gsp = np.zeros((3,16))
for i in range(1,17):
    x = np.ravel(PFT_thing['TI_INDEX'].where(PFT_thing["PCT_PFT"][i,:,:] > 0))
    m = x[~np.isnan(x)]
    g = np.multiply(np.array(PFT_thing['TI_INDEX']), np.array(PFT_thing["PCT_PFT"][i,:,:])) * 0.01
    print(np.mean(g), np.std(g), param_span[0,i-1])
    gsp[:,i-1] = np.mean(g), np.std(g), param_span[0,i-1]
    #plt.plot(param_span[0,i-1], np.mean(g)) #, yerr=np.std(g))
# %%
def count(array, value):
    count = 0
    m = array[~np.isnan(array)]
    for i in range(len(np.ravel(array))):
        if abs(np.ravel(array)[i]) < value:
            count = count + 1
    return count, len(m)

# %%
per_in_Ticutoff = np.zeros(16)
fig, axs = plt.subplots(4,4, figsize=(15,15))

for i in range(4):
    for j in range(4):
        pft = 4*i + j + 1
        if pft != 16:
            x = np.ravel(PFT_thing['TI_INDEX'].where(PFT_thing["PCT_PFT"][pft,:,:] > 0))
            a, b = count(x, 900)
            per_in_Ticutoff[pft-1] = np.round((a/b), 3)
            axs[i,j].set_title(str(4*i + j + 1) + '   ' + str(np.round((a/b), 3)))
            axs[i,j].hist(np.ravel(PFT_thing['TI_INDEX'].where(PFT_thing["PCT_PFT"][(4*i + j+1),:,:] > 00)), weights=np.ravel(0.01*PFT_thing["PCT_PFT"][(4*i + j+1),:,:]))
            axs[i,j].axvline(x=-900, color='r')
            axs[i,j].axvline(x=900, color='r')
        else:
            print(16)
# %%
def linef(a,b,x):
    return a + b*x
# based on weighted fit of QSA results (see 1504)
def s1_Ti(Ti):
    return linef(12.7151, -0.00012, Ti)

def s2_Ti(Ti):
    return linef(5.7095, -0.00367, Ti)

def m1_Ti(Ti):
    return linef(-3.45388, -0.00389, Ti)

def m2_Ti(Ti):
    return linef(48.02599, 0.00301, Ti)

# %%
# GOAL:
# Take a Ti cut off value and use it as a selection method of the map.
# For the values below the Ti, then convert Ti to the parameter
# Then across all the map, find the average of the parameter weighted by cover percentage
#
# The idea is that we can see how with different cut off Ti values,
# how does the weighted average of the parameters change. Can we find
# a relationship between this change and the PFT result (sigma, r-squared)
# %%
def Map_Ti_weight(PFT, cutoff, perc=0):
    map_PFT = PFT_thing['TI_INDEX'].where(PFT_thing["PCT_PFT"][PFT,:,:] > perc)
    map_cutoff = map_PFT.where(abs(map_PFT) < cutoff)
    total_mass = np.nansum(PFT_thing['PCT_PFT'][PFT,:,:].where(PFT_thing["PCT_PFT"][PFT,:,:] > perc)*0.01)
    map_wieght = np.multiply(np.array(map_cutoff), np.array(PFT_thing["PCT_PFT"][PFT,:,:].where(PFT_thing["PCT_PFT"][PFT,:,:] > perc))) * 0.01
    waverage = np.nansum(map_wieght) / total_mass
    return map_cutoff, map_wieght, total_mass, waverage

def nan_wmean(array, weights):
    w = np.nansum(weights)
    wmap = np.multiply(array, weights)
    return np.nansum(wmap)/w

def weight_std(weights, values):
    # https://www.automateexcel.com/stats/weighted-standard-deviation/#:~:text=Weighted%20Standard%20Deviation%20measures%20the%20spread%20of%20a,more%20weight%20than%20to%20data%20with%20less%20weight.
    wmean = nan_wmean(values, weights)
    strd = np.nansum(np.multiply(weights, (values - wmean))**2)


    return 1
# %%
# Method for adjusting cutoff and % for any paramete
cut = 1000
per = 0
refsd = np.zeros((4,16))
for i in range(1, 16):

    # make cut off map of Ti < cutoff for PFT of cover x%
    # convert Ti map to the variable map
    # Get the weighted version based on PFT cvr%
    # sum the used weights to assist with the weighted average
    # Do the weighted average

    a0 = Map_Ti_weight(i, cut, perc=per)[0]
    a1 = s1_Ti(a0)
    a2 = PFT_thing['PCT_PFT'][i,:,:].where(~np.isnan(a0))
    a3 = np.multiply(a1, a2)

    x = np.ravel(PFT_thing['TI_INDEX'].where(PFT_thing["PCT_PFT"][i,:,:] > per))
    a, b = count(x, cut)
    # weighted standard deviation

    #print(np.nanmean(a1), np.nansum(a3)/np.nansum(a2), par_spsht['s2'][i-1])
    refsd[:,i-1] = [np.nanmean(a1), np.nansum(a3)/np.nansum(a2), par_spsht['s1'][i-1], np.round((a/b), 3)]
    #a = np.nanmean(np.multiply(m1_Ti(Map_Ti_weight(i,1100)[0]), np.array(PFT_thing["PCT_PFT"][i,:,:].where(PFT_thing["PCT_PFT"][i,:,:] > 0)))*0.01)

a = plt.scatter(par_spsht['r_2'][:16], refsd[1,:]-par_spsht['s1'][:16], c=refsd[3,:], vmin=0, vmax=1)
plt.ylabel('Weighted Ti avg - PFT result (s1)')
plt.xlabel('r-squared of PFT result')
plt.colorbar(a)
# %%
PFT = 14
fig = plt.figure(figsize=(21, 11))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
a = ax.pcolormesh(lons,lats, PFT_thing['PCT_PFT'][PFT,:,:].where(PFT_thing['PCT_PFT'][PFT,:,:] > 0), transform=ccrs.PlateCarree(central_longitude=0), cmap='rainbow')
ax.set_title('PFT Distribution ' + str(PFT), fontsize=22)
ax.set_global()
ax.coastlines()
plt.colorbar(a, fraction=0.024, pad=0.01)

# %%
PFT = 1
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree(central_longitude=0))
a = ax.pcolormesh(lons, lats, m1_Ti(Map_Ti_weight(PFT,1100)[0]) - float(par_spsht['m1'][PFT-1]), transform=ccrs.PlateCarree(central_longitude=0), cmap='jet')
ax.coastlines()
ax.set_title('m1 parameter value distribution')
fig.colorbar(a, ax=ax, fraction=0.023, pad=0.04)
ay = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree(central_longitude=0))
b = ay.pcolormesh(lons, lats, s1_Ti(Map_Ti_weight(PFT,1100)[0]) - float(par_spsht['s1'][PFT-1]), transform=ccrs.PlateCarree(central_longitude=0), cmap='jet')
ay.coastlines()
ay.set_title('s1 parameter value distribution')
fig.colorbar(b, ax=ay, fraction=0.023, pad=0.04)
az = fig.add_subplot(2,2,3, projection=ccrs.PlateCarree(central_longitude=0))
c = az.pcolormesh(lons, lats, m2_Ti(Map_Ti_weight(PFT,1100)[0]) - float(par_spsht['m2'][PFT-1]), transform=ccrs.PlateCarree(central_longitude=0), cmap='jet')
az.coastlines()
az.set_title('m2 parameter value distribution')
fig.colorbar(c, ax=az, fraction=0.023, pad=0.04)
aw = fig.add_subplot(2,2,4, projection=ccrs.PlateCarree(central_longitude=0))
d = aw.pcolormesh(lons, lats, s2_Ti(Map_Ti_weight(PFT,1100)[0]) - float(par_spsht['s2'][PFT-1]), transform=ccrs.PlateCarree(central_longitude=0), cmap='jet')
aw.coastlines()
aw.set_title('s2 parameter value distribution')
fig.colorbar(d, ax=aw, fraction=0.023, pad=0.04)

# %%
pft = 1
plt.plot(np.nanmean(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360))
plt.plot(np.nanmean(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360))
plt.vlines(par_spsht['m1'][pft-1],0,360)
plt.vlines(par_spsht['m2'][pft-1],0,360)

# %%
pft = 13
fig = plt.figure(figsize=(14,5), tight_layout=True)
ax = fig.add_subplot(1,6,1)
ax.plot(np.nanmean(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
ax.fill_betweenx(np.linspace(0,360,360), np.nanmean(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
ax.axvline(par_spsht['m1'][pft-1],0,360, c='r', )
ax.grid()
ax.set_ylim([0,360])
ax.set_xlim([-18, 14]) # par_spst['m1'].describe()
ax.set_title('m1 lat mean')
ay = fig.add_subplot(1,6,2, sharey=ax)
ay.plot(np.nanmean(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
ay.fill_betweenx(np.linspace(0,360,360), np.nanmean(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
ay.axvline(par_spsht['m2'][pft-1],0,360, c='r')
ay.grid()
ay.set_ylim([0,360])
ay.set_xlim([35, 57])
ay.set_title('m2 lat mean')
az = fig.add_subplot(1,6,(3,6), projection=ccrs.PlateCarree(central_longitude=0))
a = az.contourf(lons, lats, Map_Ti_weight(pft,1100)[0], transform=ccrs.PlateCarree(central_longitude=0))
az.set_title('Distribution of ' + str(pft) + ' in Ti restrictions')
az.add_feature(cfeature.OCEAN)
az.add_feature(cfeature.LAND)
az.set_global()
az.coastlines()

# %%
pft = 1
fig = plt.figure(figsize=(14,5), tight_layout=True)
ax = fig.add_subplot(1,6,1)
ax.plot(np.nanmean(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
ax.fill_betweenx(np.linspace(0,360,360), np.nanmean(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
ax.axvline(par_spsht['s1'][pft-1],0,360, c='r', )
ax.grid()
ax.set_ylim([0,360])
ax.set_xlim([0, 26]) # par_spst['m1'].describe()
ax.set_title('s1 lat mean')
ay = fig.add_subplot(1,6,2, sharey=ax)
ay.plot(np.nanmean(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
ay.fill_betweenx(np.linspace(0,360,360), np.nanmean(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
ay.axvline(par_spsht['s2'][pft-1],0,360, c='r')
ay.grid()
ay.set_ylim([0,360])
ay.set_xlim([0, 14])
ay.set_title('s2 lat mean')
az = fig.add_subplot(1,6,(3,6), projection=ccrs.PlateCarree(central_longitude=0))
a = az.contourf(lons, lats, Map_Ti_weight(pft,1100)[0], transform=ccrs.PlateCarree(central_longitude=0))
az.set_title('Distribution of ' + str(pft) + ' in Ti restrictions')
az.add_feature(cfeature.OCEAN)
az.add_feature(cfeature.LAND)
az.set_global()
az.coastlines()
# %%
pft = 13
fig = plt.figure(figsize=(14,5), tight_layout=True)
ax = fig.add_subplot(1,6,1)
ax.plot(np.nanmean(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
ax.fill_betweenx(np.linspace(0,360,360), np.nanmean(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
ax.axvline(par_spsht['m1'][pft-1],0,360, c='r', )
ax.grid()
ax.set_ylim([0,360])
ax.set_xlim([-18, 14])
ax.set_title('m1 lat mean')
ay = fig.add_subplot(1,6,2, sharey=ax)
ay.plot(np.nanmean(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
ay.fill_betweenx(np.linspace(0,360,360), np.nanmean(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
ay.axvline(par_spsht['s1'][pft-1],0,360, c='r', )
ay.grid()
ay.set_ylim([0,360])
ay.set_xlim([0, 26])
ay.set_title('s1 lat mean')
az = fig.add_subplot(1,6,(3,6), projection=ccrs.PlateCarree(central_longitude=0))
a = az.contourf(lons, lats, Map_Ti_weight(pft,1100)[0], transform=ccrs.PlateCarree(central_longitude=0))
az.set_title('Distribution of ' + str(pft) + ' in Ti restrictions')
az.add_feature(cfeature.OCEAN)
az.add_feature(cfeature.LAND)
az.set_global()
az.coastlines()
# %%
pft = 13
fig = plt.figure(figsize=(14,5), tight_layout=True)
ax = fig.add_subplot(1,6,1)
ax.plot(np.nanmean(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
ax.fill_betweenx(np.linspace(0,360,360), np.nanmean(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
ax.axvline(par_spsht['m2'][pft-1],0,360, c='r')
ax.grid()
ax.set_ylim([0,360])
ax.set_xlim([35, 57])
ax.set_title('m2 lat mean')
ay = fig.add_subplot(1,6,2, sharey=ax)
ay.plot(np.nanmean(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
ay.fill_betweenx(np.linspace(0,360,360), np.nanmean(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
ay.axvline(par_spsht['s2'][pft-1],0,360, c='r')
ay.grid()
ay.set_ylim([0,360])
ay.set_xlim([0, 14])
ay.set_title('s2 lat mean')
az = fig.add_subplot(1,6,(3,6), projection=ccrs.PlateCarree(central_longitude=0))
a = az.contourf(lons, lats, Map_Ti_weight(pft,1100)[0], transform=ccrs.PlateCarree(central_longitude=0))
az.set_title('Distribution of ' + str(pft) + ' in Ti restrictions')
az.add_feature(cfeature.OCEAN)
az.add_feature(cfeature.LAND)
az.set_global()
az.coastlines()
# %%


# %%
# Making Figure 1 based on PSIIClim (Ordered_set)
from cartopy import feature as cfeature
lts = np.zeros(len(np.unique(Ordered_set['latlon'])))
lns = np.zeros(len(np.unique(Ordered_set['latlon'])))
for i in range(0,len(np.unique(Ordered_set['latlon']))):
    lts[i] = float(np.unique(Ordered_set['latlon'])[i].split(',')[0])
    lns[i] = float(np.unique(Ordered_set['latlon'])[i].split(',')[1])

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree(central_longitude=0))
ax.scatter(lns, lts, s=70, c='r', edgecolor='black', transform=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND)
ax.set_global()
ax.coastlines()
# %%
# Figure 2 is made out of ResearchPlot line 818

# Making of Figure 3 based on par_spsht
tot_tol = (par_spsht['m2'] - 2*par_spsht['s2']) - (par_spsht['m1'] + 2*par_spsht['s1'])
tot_res = par_spsht['s1'] + par_spsht['s2']


plt.scatter(tot_res[:16], tot_tol[:16])
# %%
###########################################################

# Working with CMIP6 data

###########################################################
import xarray as xr

lats = PFT_thing['LAT']
lons = PFT_thing['LON']

tcmip6_1 = xr.load_dataset('tas_day_CESM2_ssp245_r4i1p1f1_gn_20700101-20820521_v20200528.nc')
tcmip6_2 = xr.load_dataset('tas_day_CESM2_ssp245_r4i1p1f1_gn_20820522-20941009_v20200528.nc')
tcmip6_3 = xr.load_dataset('tas_day_CESM2_ssp245_r4i1p1f1_gn_20941010-21001231_v20200528.nc')

tcmip6 = xr.concat([tcmip6_1,tcmip6_2,tcmip6_3], dim='time')
# %%
###############################################

# NOTE THAT BOTH crrtmulti AND tcmip6 ARE NOT ALIGNED WITH
# THE PFT_thing LAT & LON

###############################################

# Map of previous 30 year climatology
a = plt.pcolormesh(crrtmulti[:,:,6])
plt.colorbar(a)
# %%
# Map of new 30 year climatology
b = plt.pcolormesh(tcmip6.mean(dim='time')['tas'])
plt.colorbar(b)
# %%
# Example taken from https://stackoverflow.com/questions/42275304/changing-data-resolution-in-python

from scipy.interpolate import RegularGridInterpolator

def regrid(data, out_x, out_y):
    m = max(data.shape[0], data.shape[1])
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data)

    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y), np.linspace(0, 1.0/m, out_x))

    return interpolating_function((xv, yv))
# %%
# Map of difference (b - a)
c = plt.pcolormesh(tcmip6.mean(dim='time')['tas'] - regrid(crrtmulti[:,:,6],192,288), vmin=-20, vmax=20, cmap='seismic')
plt.colorbar(c)
# %%
# Regriding the CMIP6 data
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree(central_longitude=0))
a = ax.pcolormesh(lons, lats, np.roll(regrid(np.array(tcmip6.mean(dim='time')['tas']), 360, 720) - crrtmulti[:,:,6], 360), vmin=-20, vmax=20, cmap='seismic', transform=ccrs.PlateCarree(central_longitude=0))
ax.set_global()
ax.coastlines()
cbar = plt.colorbar(a, ax=ax, fraction=0.023, pad=0.01)
cbar.ax.get_yaxis().labelpad = 36
cbar.ax.set_ylabel('(CMIP6 - CRUNCEP) climatology \n temperature difference [C]', rotation=270, fontsize=16)
# %%
# Trying to generate the map of where PFT dominate
# Made changes to PFT_thing to make this easier
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree(central_longitude=0))
a = ax.pcolormesh(lons, lats, PFT_thing['MAX_PFT'].where(PFT_thing['LANDMASK'] == 1), transform=ccrs.PlateCarree())
ax.coastlines()
fig.colorbar(a, ax=ax, fraction=0.023, pad=0.04)
# %%
# Need a method to convert the 4 unusable PFTs to nan on map (after T_MH established)
bad = [0,3,8,11,12]
GPFT_thing = PFT_thing
for i in bad:
    print(i)
    GPFT_thing['PCT_PFT'][i,:,:] = np.nan * GPFT_thing['PCT_PFT'][i,:,:]

GPFT_thing['GMAX_PFT'] = (['lat', 'lon'], np.nanargmax(np.array(GPFT_thing['PCT_PFT'][:,:,:]), axis=0))
# %%
# For some reason to keep the masking working, I need to reinit the PFT_thing file
PFT_thing = xr.load_dataset('c:/Users/pjneri/Desktop/mksrf_pft_0.5x0.5_simyr2005.c090313.nc')
PFT_thing["TI_INDEX"] = (['lat', 'lon'], adj_glob_Ti)
# %%
from matplotlib.colors import ListedColormap
cmap = ListedColormap(
['green', 'plum', 'olive', 'chartreuse', 'silver', 'aqua', 'orangered', 
'tan', 'teal', 'sienna', 'yellow', 'yellowgreen', 'darkgreen', 'darkblue', 'indigo','red']) #, 'indigo', 'yellowgreen', 'fuchsia', 'crimson', 'wheat', 'plum'

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree(central_longitude=0))
a = ax.pcolormesh(lons, lats, GPFT_thing['GMAX_PFT'].where(GPFT_thing['LANDMASK'] == 1).where(GPFT_thing['GMAX_PFT'] != 0).where(PFT_thing['PCT_PFT'][0,:,:] != 100), transform=ccrs.PlateCarree(), cmap=cmap, vmax=16)
ax.coastlines()

cbar = plt.colorbar(a, ax=ax, fraction=0.023, pad=0.01)
# https://stackoverflow.com/questions/15908371/matplotlib-colorbars-and-its-text-labels
cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(['$NET-Te$','$NET-Bo$','$NDT-Bo$','$BET-Tr$','$BET-Te$','$BDT-Tr$','$BDT-Te$','$BDT-Bo$','$BES$','$BDS-Te$','$BDS-Bo$','$C3-AG$','$C3-NAG$','$C4-G$','$C3-C$','$C4-C$']):
    cbar.ax.text(18, 15*( j + 1.33)/16, lab, ha='left')

for j, lab in [(3,'X'),(8,'X'),(11,'X'),(12,'X'),(16,'X')]:
    cbar.ax.text(8, 15*(j + 0.5)/16, lab, ha='center', va='center', fontsize=21)

cbar.ax.get_yaxis().labelpad = 73
cbar.ax.set_ylabel('Maximum Percent Coverage PFT', rotation=270, fontsize=16)
# %%
##################
# DROPPED
##################

# Creating a new CMIP6 Ti index
Caat = tcmip6.mean(dim='time')['tas']
# Need way to select values in set months
from datetime import datetime
cmiptime = tcmip6.indexes['time'].to_datetimeindex()
ncmiptime = pd.Series()
for i in range(len(cmiptime)):
    ncmiptime = pd.concat([ncmiptime, pd.Series([datetime.strptime(str(cmiptime[i]),'%Y-%m-%d %H:%M:%S')])], ignore_index=True)

# Csmet = 
# Cwmet = 

# %%
# Figure 5-5 plots
pft = 1
fig = plt.figure(figsize=(14,5), tight_layout=True)
fig.suptitle('PFT ' + str(pft), fontsize=20)
fig.supylabel('Latitude (' + '\u00B0' + ')', fontsize=15)
ax = fig.add_subplot(1,5,1)
ax.plot(np.nanmean(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
ax.fill_betweenx(np.linspace(0,360,360), np.nanmean(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(m1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
ax.axvline(par_spsht['m1'][pft-1],0,360, c='r')
ax.grid()
ax.set_ylim([0,360])
ax.set_xlim([-23, 7])
ax.set_title('m1 parameter', fontsize=13)
ay = fig.add_subplot(1,5,2, sharey=ax)
ay.plot(np.nanmean(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
ay.fill_betweenx(np.linspace(0,360,360), np.nanmean(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(s1_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
ay.axvline(par_spsht['s1'][pft-1],0,360, c='r')
ay.grid()
ay.set_ylim([0,360])
ay.set_xlim([0, 25])
ay.set_title('s1 parameter', fontsize=13)
az = fig.add_subplot(1,5,5, sharey=ax)
az.plot(np.nanmean(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
az.fill_betweenx(np.linspace(0,360,360), np.nanmean(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(m2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
az.axvline(par_spsht['m2'][pft-1],0,360, c='r')
az.grid()
az.set_ylim([0,360])
az.set_xlim([35, 57])
az.set_title('m2 parameter', fontsize=13)
aw = fig.add_subplot(1,5, 4, sharey=ax)
aw.plot(np.nanmean(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.linspace(0,360,360), color='black')
aw.fill_betweenx(np.linspace(0,360,360), np.nanmean(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) - np.nanstd(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), np.nanmean(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1) + np.nanstd(s2_Ti(Map_Ti_weight(pft,1100)[0]), axis=1), color='black',alpha=0.2)
aw.axvline(par_spsht['s2'][pft-1],0,360, c='r')
aw.grid()
aw.set_ylim([0,360])
aw.set_xlim([0, 14])
aw.set_title('s2 parameter', fontsize=13)
# Need to perform unit conversion to km covered
# delx = 2 pi r_earth cos(lat) (dellon / 360)
# dely = 2 pi r_earth (dellat / 360)
# WANT:
# (area(lat) * %cover) / (area(lat) * 100%cover)
al = fig.add_subplot(1,5,3, sharey=ax)
al.plot(np.nansum(PFT_thing['PCT_PFT'][pft,:,:].where(PFT_thing['LANDMASK'] == 1), axis=1), np.linspace(0,360,360), color='black', label='All grid % coverage sum')
al.plot(np.nansum(PFT_thing['PCT_PFT'][pft,:,:].where(PFT_thing['LANDMASK'] == 1).where(abs(PFT_thing['TI_INDEX']) < 1100), axis=1), np.linspace(0,360,360), color='red', label='Ti-cutoff contributing grid %')
al.grid(True)
al.set_title('Ti allowed contribution')
# %%
# 6-14 Adjustment to the 5-5 concept, and using the ad_glob_Ti (newmean)
used_plants = [1,2,3,5,7,9,10,13,15] # selected based on above 50% (8,11,12 dropped for PFT fit)
colors = []

fig = plt.figure(figsize=(10,10), tight_layout=True)
ax = fig.add_subplot(2,2,1)
ay = fig.add_subplot(2,2,2)
az = fig.add_subplot(2,2,3)
aw = fig.add_subplot(2,2,4)
for i in used_plants:
    pft = i
    ax.plot(np.nanmean(m1_Ti(Map_Ti_weight(pft,900)[0]) + 2*s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - (par_spsht['m1'][pft-1] + 2*par_spsht['s1'][pft-1]), np.linspace(0,360,360), label=str(pft))
    ax.grid(True)
    ax.set_ylim([0,360])
    ax.set_title('$T_{MC}$', fontsize=16)
    #ax.legend()
    ay.plot(np.nanmean(m2_Ti(Map_Ti_weight(pft,900)[0]) - 2*s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - (par_spsht['m2'][pft-1] - 2*par_spsht['s2'][pft-1]), np.linspace(0,360,360), label=str(pft))
    ay.grid(True)
    ay.set_ylim([0,360])
    ay.set_title('$T_{MH}$', fontsize=16)
    #ay.legend()
    az.plot(np.nanmean(s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - par_spsht['s1'][pft-1], np.linspace(0,360,360), label=str(pft))
    az.grid(True)
    az.set_ylim([0,360])
    az.set_title('$s_{1}$', fontsize=16)
    #az.legend()
    aw.plot(np.nanmean(s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - par_spsht['s2'][pft-1], np.linspace(0,360,360), label=str(pft))
    aw.grid(True)
    aw.set_ylim([0,360])
    aw.set_title('$s_{2}$', fontsize=16)
    aw.legend()



# %%
# Not sure if this is perhaps what we want instead (sum versus mean??)
pft = 13
fig = plt.figure(figsize=(3,5), tight_layout=True)
al = fig.add_subplot(1,1,1)
al.plot(np.nanmean(Map_Ti_weight(pft, 1100)[0], axis=1), np.linspace(0,360,360), color='black')
al.plot(np.nanmean(Map_Ti_weight(pft, 10000)[0], axis=1), np.linspace(0,360,360), color='red', linestyle='dashed')
al.grid(True)
# %%
c_par_spsht = sort_par # See section on tol-res trade off

slp1, cept1, r1, p1, se1 = stt.linregress(c_par_spsht['T_MC'] - np.mean(c_par_spsht['T_MC']), c_par_spsht['s1'])
slp2, cept2, r2, p2, se2 = stt.linregress(c_par_spsht['T_MH'] - np.mean(c_par_spsht['T_MH']), c_par_spsht['s2'])

x1 = np.linspace(-20, 20, 60)
x2 = np.linspace(-20, 20, 60)
fig = plt.figure(figsize=(8,4.5), tight_layout=True)
fig.suptitle('Tolerance - Resilience Tradeoffs in PFTs', fontsize=17)
ax = fig.add_subplot(1,2,1)
ax.set_ylim([0,26])
ax.set_xlim([-20,20])
ax.grid(True)
ax.set_title('Cold temperature', fontsize=15)
ax.set_ylabel('Resiliency parameter (s1)', fontsize=13)
ax.set_xlabel('Deviation from mean ' + r'$T_{MC}$', fontsize=13)
ax.scatter(c_par_spsht['T_MC'][:] - np.mean(c_par_spsht['T_MC']), c_par_spsht['s1'][:], color='blue') 
ax.plot(x1, linef(cept1,slp1,x1), color='cyan', label='y = 9.94 + 0.45 x\n' + r'$r^{2}$' + ' = 0.57')
ax.legend()
ay = fig.add_subplot(1,2,2, sharey=ax, sharex=ax)
ay.grid(True)
ay.set_title('Hot temperature', fontsize=15)
ay.set_ylabel('Resiliency parameter (s2)', fontsize=13)
ay.set_xlabel('Deviation from mean ' + r'$T_{MH}$', fontsize=13)
ay.scatter(c_par_spsht['T_MH'][:] - np.mean(c_par_spsht['T_MH']), c_par_spsht['s2'][:], color='red') 
ay.plot(x2, linef(cept2,slp2,x2), color='orange', label='y = 7.26 - 0.47 x\n' + r'$r^{2}$' + ' = 0.69')
ay.legend()
# %%
t_offset1 = GPFT_thing['GMAX_PFT'].where(GPFT_thing['LANDMASK'] == 1).where(GPFT_thing['GMAX_PFT'] != 0).where(PFT_thing['PCT_PFT'][0,:,:] != 100)
t_offset2 = GPFT_thing['TI_INDEX'].where(GPFT_thing['LANDMASK'] == 1).where(GPFT_thing['GMAX_PFT'] != 0).where(PFT_thing['PCT_PFT'][0,:,:] != 100).where(abs(GPFT_thing['TI_INDEX']) < 900)
#t_offset3 = PFT_thing['TI_INDEX'].where(PFT_thing['LANDMASK'] == 1).where(abs(PFT_thing["TI_INDEX"]) < 1100)

def Methd_of_TMH_conv1(dataset):
    array = np.array(dataset)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if np.isnan(array[i,j]) == True:
                array[i,j] = np.nan
            else:
                array[i,j] = par_spsht['m2'][array[i,j] - 1] - 2*par_spsht['s2'][array[i,j] - 1]
    return array

def Methd_of_TMH_conv2(dataset):
    array = np.array(dataset)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if np.isnan(array[i,j]) == True:
                array[i,j] = np.nan
            else:
                array[i,j] = m2_Ti(array[i,j]) - 2*s2_Ti(array[i,j])
    return array
# %%
fig = plt.figure(figsize=(20, 10), tight_layout=True)
ax = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree(central_longitude=0))
a = ax.pcolormesh(lons, lats, Methd_of_TMH_conv1(t_offset1), transform=ccrs.PlateCarree(), cmap='viridis')
ax.coastlines()
ax.set_title('PFT Max (GPFT) informed ' + r'$T_{MH}$')
cbar = plt.colorbar(a, ax=ax, fraction=0.023, pad=0.01)

ay = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree(central_longitude=0))
a = ay.pcolormesh(lons, lats, Methd_of_TMH_conv2(t_offset2), transform=ccrs.PlateCarree(), cmap='viridis')
ay.coastlines()
ay.set_title('CTI (900 limit) informed ' + r'$T_{MH}$')
cbar2 = plt.colorbar(a, ax=ay, fraction=0.023, pad=0.01)

# %%
def point_value(array, x):
    narray = np.array(array)
    nx = np.array(x)
    new_array = np.zeros((array.shape[0], array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if np.isnan(narray[i,j]) == True:
                new_array[i,j] = np.nan
            else:
                pft = int(narray[i,j] - 1) # Assuming GPFT_Thing GMAX_PFT
                a, c1, c2, s1, s2 = par_spsht.iloc[pft, 1:6][:]
                Aa1 = (nx[i,j] - c1)/s1
                Aaa1 = math.erf(Aa1)
                Aa2 = -(nx[i,j] - c2)/s2
                Aaa2 = math.erf(Aa2)
                Ayy = (a/2)* (Aaa1 + Aaa2)
                new_array[i,j] = Ayy
    return new_array

def boint_value(array, x):
    narray = np.array(array)
    nx = np.array(x)
    new_array = np.zeros((array.shape[0], array.shape[1])) 
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):   
            if np.isnan(narray[i,j]) == True:
                new_array[i,j] = np.nan
            else:
                a = 0.80
                c1 = m1_Ti(narray[i,j])
                c2 = m2_Ti(narray[i,j])
                s1 = s1_Ti(narray[i,j])
                s2 = s2_Ti(narray[i,j])
                Aa1 = (nx[i,j] - c1)/s1
                Aaa1 = math.erf(Aa1)
                Aa2 = -(nx[i,j] - c2)/s2
                Aaa2 = math.erf(Aa2)
                Ayy = (a/2)* (Aaa1 + Aaa2)
                new_array[i,j] = Ayy         
    return new_array   

# %%
ofset = np.roll(regrid(np.array(tcmip6.mean(dim='time')['tas']), 360, 720) - crrtmulti[:,:,6], 360)
newt = Methd_of_TMH_conv1(t_offset1) + ofset
bewt = Methd_of_TMH_conv2(t_offset2) + ofset

fMH = point_value(t_offset1, Methd_of_TMH_conv1(t_offset1))
nfMH = point_value(t_offset1, newt)

bMH = boint_value(t_offset2, Methd_of_TMH_conv2(t_offset2))
nbMH = boint_value(t_offset2, bewt)

finalplt = np.zeros((360,720))
binalplt = np.zeros((360,720))
for i in range(360):
    for j in range(720):
        finalplt[i,j] = (fMH[i,j] - nfMH[i,j])/fMH[i,j]
        binalplt[i,j] = (bMH[i,j] - nbMH[i,j])/bMH[i,j]

# %%
fig = plt.figure(figsize=(20, 10), tight_layout=True)
ay = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree(central_longitude=0))
a = ay.pcolormesh(lons, lats, finalplt, vmin=0, vmax=1,transform=ccrs.PlateCarree(), cmap='jet')
ay.coastlines()
ay.set_title('PFT % decline')
cbar2 = plt.colorbar(a, ax=ay, fraction=0.023, pad=0.01)
ax = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree(central_longitude=0))
b = ax.pcolormesh(lons, lats, binalplt, vmin=0, vmax=1,transform=ccrs.PlateCarree(), cmap='jet')
ax.coastlines()
ax.set_title('CTI % decline')
cbar2 = plt.colorbar(b, ax=ax, fraction=0.023, pad=0.01)
# %%
GPFT_thing["Final"] = (['lat', 'lon'], finalplt)
GPFT_thing["Binal"] = (['lat', 'lon'], binalplt)
GPFT_thing['Devi'] = (['lat', 'lon'], ofset)
# %%
x = np.linspace(-4000,3000,7000)
plt.title('CTI parameter equations')
plt.plot(x, m1_Ti(x) + 2*s1_Ti(x), label='$T_{MC}$')
plt.plot(x, m1_Ti(x), c='cyan', label='$m_{1}$')
plt.plot(x, m2_Ti(x) - 2*s2_Ti(x), label='$T_{MH}$')
plt.plot(x, m2_Ti(x), c='red', label='$m_{2}$')
plt.vlines([-900,900],-20,65, color='k', label='TI bounds')
plt.xlim([-4000,3000])
plt.xlabel('CTI value')
plt.ylim([-20,65])
plt.ylabel('Temperature (C)')
plt.scatter(np.ravel(GPFT_thing['TI_INDEX']), np.ravel(bewt), marker='+', alpha=0.3, label='Deviation adj. \n $T_{MH}$')
plt.legend(loc='upper left')
# %%
fig = plt.figure(figsize=(7, 8))
ay = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree(central_longitude=0))
a = ay.contourf(lons, lats, finalplt - binalplt, vmin=-1, vmax=1, transform=ccrs.PlateCarree(), cmap='brg')
ay.coastlines()
ay.set_title('Diff. % decline (PFT - CTI)')
cbar2 = plt.colorbar(a, ax=ay, fraction=0.023, pad=0.01)
ax = fig.add_subplot(2,1,2, aspect='equal')
ax.scatter(np.ravel(finalplt), np.ravel(binalplt))
ax.set_xlabel('PFT % decline')
ax.set_ylabel('CTI % decline')
ax.grid()
ax.plot(np.linspace(0,1,2),np.linspace(0,1,2),'k')
# %%
fig = plt.figure(figsize=(14, 5))
ay = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree(central_longitude=0))
a = ay.pcolormesh(lons, lats, finalplt - binalplt, transform=ccrs.PlateCarree(), cmap='seismic')
ay.coastlines()
ay.set_title('Diff. % decline (PFT - CTI)')

ay.add_feature(cfeature.LAND)
cbar2 = plt.colorbar(a, ax=ay, fraction=0.023, pad=0.01, extend='both')
# %%
plt.plot(lats, np.nanmean(finalplt, axis=1)) # PFT % decline
plt.plot(lats, np.nanmean(binalplt, axis=1)) # CTI % decline
# %%
GPFT_thing['DIFF'] = (['lat', 'lon'], finalplt - binalplt)

fig = plt.figure(figsize=(14, 8))
ay = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree(central_longitude=0))
a = ay.pcolormesh(lons, lats, GPFT_thing['DIFF'].where(GPFT_thing['DIFF'] < -0.3), transform=ccrs.PlateCarree(), cmap='jet')
ay.coastlines()
ay.set_title('')
cbar2 = plt.colorbar(a, ax=ay, fraction=0.023, pad=0.01)
# %%
plt.scatter(GPFT_thing['Final'].where(GPFT_thing['DIFF'] > 0.3), GPFT_thing['Binal'], c=GPFT_thing['Devi'])
plt.colorbar()
plt.plot(np.linspace(0,1,2),np.linspace(0,1,2),'k')
# %%
# plt.plot(np.count_nonzero(~np.isnan(PFT_thing['TI_INDEX'].where(PFT_thing['LANDMASK'] == 1).where(abs(PFT_thing['TI_INDEX']) < 900)), axis=1))

fig = plt.figure(figsize=(21,6), tight_layout=True)
ax = fig.add_subplot(1,7,(1,2))
ax.plot(np.mean(PFT_thing['TI_INDEX'], axis=1), np.linspace(-90,90,360), color='black')
ax.vlines([-900, 900], -90, 90)
ax.fill_betweenx(np.linspace(-90,90,360),-900,900, alpha=0.4, color='r', where=abs(np.mean(PFT_thing['TI_INDEX'], axis=1)) < 900)
ax.grid()
ax.set_xlabel('CTI')
ax.set_ylim([-90,90])
#ax.set_xlim([0, 26]) # par_spst['m1'].describe()
ax.set_xscale('symlog')
ax.set_title('CTI lat mean')
az = fig.add_subplot(1,7,(3,7), projection=ccrs.PlateCarree(central_longitude=0))
a = az.pcolormesh(lons, lats, PFT_thing['TI_INDEX'], cmap='seismic', vmin=-4000, vmax=4000, transform=ccrs.PlateCarree(central_longitude=0))
az.contourf(lons,lats, PFT_thing['TI_INDEX'].where((890 > abs(PFT_thing['TI_INDEX'])) | (abs(PFT_thing['TI_INDEX']) < 910)), colors='black', hatches=['/'], levels=1,alpha=0, transform=ccrs.PlateCarree(central_longitude=0))
az.set_title('Distribution of CTI')
az.add_feature(cfeature.OCEAN)
az.add_feature(cfeature.LAND)
az.set_global()
az.coastlines()
plt.colorbar(a, ax=az, extend='both', label='CTI')
# %%
# 6-13-22 Final to-do list

# Check and compare the global cti mean values to the location cti mean values
# for the different climate indicies (check!)

# Convert the 5 x 7 lat mean plot to a multi-line 2x2 parameter plot
# (potentially do the tolerance instead of the m params)
# Then remove the distribution portion and make an appendix figure (check!)

# Convert the grid size to km x km scale (ignore)

# Adjust vmin, vmax on difference of clim. plot

# Create plot for the PFT, CTI comparison and decline histograms

# %%
# NEED TO COVERT TO A SCATTER PLOT OF TOTAL RESILIENCE TO TOLERANCE RANGE

# 6-14 Adjustment to the 5-5 concept, and using the ad_glob_Ti (newmean)
used_plants = [1,2,5,7,9,10,13,15] # selected based on above 50% (8,11,12 dropped for PFT fit)
markers = ['.', '+', 'o', 'H', 'v', 's', '*', 'd']

fig = plt.figure(figsize=(15,10), tight_layout=True)
ax = fig.add_subplot(2,3,1)
ay = fig.add_subplot(2,3,4)
az = fig.add_subplot(2,3,(2,5))
aw = fig.add_subplot(2,3,(3,6), sharey=az)
ax.axhline(0, color='k')
ax.axvline(0, color='k')
ay.axhline(0, color='k')
ay.axvline(0, color='k')
ax.set_title('Cold temperature\n response deviations')
ax.set_title('a)', loc='left', fontsize=15)
ax.set_xlabel('TMC Tolerance deviations')
ax.set_ylabel('S1 Resilience deviations')
ay.set_title('Hot temperature\n response deviations')
ay.set_title('b)', loc='left', fontsize=15)
ay.set_xlabel('TMH Tolerance deviations')
ay.set_ylabel('S2 Resilience deviations')
az.set_title('Tolerance Range ($T_{MH}$ - $T_{MC}$)\n across latitudes')
az.set_title('c)', loc='left', fontsize=15)
az.set_xlabel('Range [deg C]')
az.set_ylabel('Latitude grid number (0.5 degree resolution)')
aw.set_title('Total Tolerance deviation\n acorss latitudes')
aw.set_title('d)', loc='left', fontsize=15)
aw.set_xlabel('Deviation')
for i in range(8):
    pft = used_plants[i]
    ax.scatter(
        #np.nanmean(m2_Ti(Map_Ti_weight(pft,900)[0]) - 2*s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - (par_spsht['m2'][pft-1] - 2*par_spsht['s2'][pft-1]), #+
        np.nanmean(m1_Ti(Map_Ti_weight(pft,900)[0]) + 2*s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - (par_spsht['m1'][pft-1] + 2*par_spsht['s1'][pft-1]),
        np.nanmean(s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - par_spsht['s1'][pft-1],
        #np.nanmean(s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - par_spsht['s2'][pft-1],        
        #c=np.linspace(0,360,360),
        marker=markers[i],
        label=str(pft)
    )
    ay.scatter(
        np.nanmean(m2_Ti(Map_Ti_weight(pft,900)[0]) - 2*s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - (par_spsht['m2'][pft-1] - 2*par_spsht['s2'][pft-1]),# +
        #np.nanmean(m1_Ti(Map_Ti_weight(pft,900)[0]) + 2*s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - (par_spsht['m1'][pft-1] + 2*par_spsht['s1'][pft-1]),
        #np.nanmean(s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - par_spsht['s1'][pft-1] +
        np.nanmean(s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - par_spsht['s2'][pft-1],        
        #c=np.linspace(0,360,360),
        marker=markers[i],
        label=str(pft)
    )
    az.plot(
        np.nanmean(m2_Ti(Map_Ti_weight(pft,900)[0]) - 2*s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) -
        np.nanmean(m1_Ti(Map_Ti_weight(pft,900)[0]) + 2*s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1),
        np.linspace(-90,90,360),
        label=str(pft)
    )
    az.fill_betweenx(
        np.linspace(-90,90,360),
        (np.nanmean(m2_Ti(Map_Ti_weight(pft,900)[0]) - 2*s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) -
        np.nanmean(m1_Ti(Map_Ti_weight(pft,900)[0]) + 2*s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1)) -
        (np.nanstd(m2_Ti(Map_Ti_weight(pft,900)[0]) - 2*s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) +
        np.nanstd(m1_Ti(Map_Ti_weight(pft,900)[0]) + 2*s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1)),
        (np.nanmean(m2_Ti(Map_Ti_weight(pft,900)[0]) - 2*s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) -
        np.nanmean(m1_Ti(Map_Ti_weight(pft,900)[0]) + 2*s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1)) +
        (np.nanstd(m2_Ti(Map_Ti_weight(pft,900)[0]) - 2*s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) +
        np.nanstd(m1_Ti(Map_Ti_weight(pft,900)[0]) + 2*s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1)),
        alpha=0.1
    )
    az.legend()
    aw.plot(
        (np.nanmean(m2_Ti(Map_Ti_weight(pft,900)[0]) - 2*s2_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - (par_spsht['m2'][pft-1] - 2*par_spsht['s2'][pft-1])) -
        (np.nanmean(m1_Ti(Map_Ti_weight(pft,900)[0]) + 2*s1_Ti(Map_Ti_weight(pft,900)[0]), axis=1) - (par_spsht['m1'][pft-1] + 2*par_spsht['s1'][pft-1])),
        np.linspace(-90,90,360)
    )
ax.legend()
# %%
# Finding % of land surface that is covered by | CTI | < 900
r_earth = 6371000 # [m]

latbands = 2 * np.pi * r_earth * np.cos(np.deg2rad(np.linspace(-89.75, 89.75, 360))) / 720

total_surf_grids = np.count_nonzero(~np.isnan(PFT_thing['TI_INDEX'].where(PFT_thing['LANDMASK'] == 1)), axis=1)
CTI_cut_surf_gds = np.count_nonzero(~np.isnan(PFT_thing['TI_INDEX'].where(PFT_thing['LANDMASK'] == 1).where(abs(PFT_thing['TI_INDEX']) < 900)), axis=1)
perct_grids_cont = CTI_cut_surf_gds / total_surf_grids

plt.plot(np.count_nonzero(~np.isnan(PFT_thing['TI_INDEX'].where(PFT_thing['LANDMASK'] == 1).where(abs(PFT_thing['TI_INDEX']) < 900)), axis=1))
plt.plot(np.count_nonzero(~np.isnan(PFT_thing['TI_INDEX'].where(PFT_thing['LANDMASK'] == 1)), axis=1))
# %%
plt.plot(np.multiply(total_surf_grids * latbands * np.max(latbands), perct_grids_cont))
plt.plot(total_surf_grids * latbands * np.max(latbands))
# %%
gctimap = PFT_thing['TI_INDEX'].where(PFT_thing['LANDMASK'] == 1)
sctimap = np.nanstd(PFT_thing['TI_INDEX'].where(PFT_thing['LANDMASK'] == 1), axis=1)
plt.plot(
    np.nanmean((m2_Ti(gctimap) - 2*s2_Ti(gctimap)) -
    (m1_Ti(gctimap) + 2*s1_Ti(gctimap)), axis=1),
    np.linspace(-90,90,360)
)
plt.xlim([0,70])
# %%
fig, axs = plt.subplots(4,4, figsize=(15,15))
for i in range(4):
    for j in range(4):
        pft = 4*i + j + 1
        if pft not in [3, 8, 11, 12, 16]:
            axs[i,j].set_title(str(4*i + j + 1))
            axs[i,j].set_xlim([0,1])
            axs[i,j].grid(True)
            axs[i,j].hist((np.ravel(GPFT_thing['Final'].where(GPFT_thing['GMAX_PFT'] == pft)),np.ravel(GPFT_thing['Binal'].where(GPFT_thing['GMAX_PFT'] == pft))), color=['yellowgreen','blue'], density=False)
            axs[i,j].axvline(x=np.nanquantile(np.ravel(GPFT_thing['Final'].where(GPFT_thing['GMAX_PFT'] == pft)), 0.95), color='k', linestyle='dashed', label='PFT decline 95%tile')
            axs[i,j].axvline(x=np.nanquantile(np.ravel(GPFT_thing['Binal'].where(GPFT_thing['GMAX_PFT'] == pft)), 0.95), color='k', label='CTI decline 95%tile')
            axs[i,j].axvline(x=np.nanmax(np.ravel(GPFT_thing['Final'].where(GPFT_thing['GMAX_PFT'] == pft))), color='yellowgreen', label='PFT decline max')
            axs[i,j].axvline(x=np.nanmax(np.ravel(GPFT_thing['Binal'].where(GPFT_thing['GMAX_PFT'] == pft))), color='blue', label='CTI decline max')
            if pft==4:
                axs[i,j].legend()
        else:
            axs[i,j].set_title(str(4*i + j + 1))
            axs[i,j].axline((0, 0), (1, 1), color='k')
            axs[i,j].axline((0, 1), (1, 0), color='k')

# %%
fig = plt.figure(figsize=(9,3), tight_layout=True)
ax = fig.add_subplot(1,2,1)
ay = fig.add_subplot(1,2,2)
fig.suptitle('Distribution of % decline between both methods')
ax.hist([np.ravel(finalplt),np.ravel(binalplt)], density=True, bins=25, color=['yellowgreen','blue'], label=['PFT decline %','CTI decline %'], histtype='bar')
ax.set_ylabel('Normalized grid count')
ax.legend()
ay.hist([np.ravel(finalplt),np.ravel(binalplt)], density=False, bins=25, color=['yellowgreen','blue'],histtype='bar')
ay.set_yscale('log')
ay.set_ylabel('Raw grid count')
# %%
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree(central_longitude=0))
a = ax.pcolormesh(lons, lats, PFT_thing['PCT_PFT'][9,:,:].where(PFT_thing['PCT_PFT'][9,:,:] > 0), transform=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.STATES)
fig.colorbar(a, ax=ax, fraction=0.023, pad=0.04)
# %%
