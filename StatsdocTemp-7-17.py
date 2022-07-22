# Attempting a 2-way ANOVA using pingouin 
# working in hastools2
# %%
from numpy.core.function_base import linspace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
import math

# %%
def residualz(x, y, params):
        A, m1, s1, m2, s2 = params[:]  
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
        return resid


def ANOVAattempt(dataset, model=None):
    """Model column is for params if residual check is the goal"""
    if model == None:
        dCont = {'FvFm': dataset['phiPSIImax'], 'Temp': dataset['temptemp'],
          'Age': dataset['Age (d)'], 'PFT': dataset['PFT #'],
          'Type': dataset['type'], 'Time': dataset['timetime'],
          'CladeA': dataset['Clade A'], 'CladeB': dataset['Clade B'],
          'Order': dataset['Order'], 'Family': dataset['Family']}
        DfCont = pd.DataFrame(data=dCont)

        model = ols('FvFm ~ C(Time) + C(Temp) + C(Type) + C(Time):C(Temp) + C(Time):C(Type)\
            + C(Temp):C(Type) + C(Time):C(Temp):C(Type)', data=DfCont).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)
        print(anova_table)
    else:
        dataset['resid'] = residualz(dataset['HeatMid'], dataset['phiPSIImax'], model)
        dCont = {'FvFm': dataset['phiPSIImax'], 'Temp': dataset['temptemp'],
          'Age': dataset['Age (d)'], 'PFT': dataset['PFT #'],
          'Type': dataset['type'], 'Time': dataset['timetime'],
          'CladeA': dataset['Clade A'], 'CladeB': dataset['Clade B'],
          'Order': dataset['Order'], 'Family': dataset['Family'],
          'resid': dataset['resid']}
        DfCont = pd.DataFrame(data=dCont)
        choice = input('Selecting a level of Genetic Division')
        if choice == 'model1':
            model = ols('resid ~ C(Time) + C(CladeA) + C(Time):C(CladeA)', data=DfCont).fit()
        elif choice == 'model2':
            model = ols('resid ~ C(Time) + C(CladeB) + C(Time):C(CladeB)', data=DfCont).fit()
        elif choice == 'model3':
            model = ols('resid ~ C(Time) + C(Order) + C(Time):C(Order)', data=DfCont).fit()
        elif choice == 'model4':
            model = ols('resid ~ C(Time) + C(Family) + C(Time):C(Family)', data=DfCont).fit()

        anova_table = sm.stats.anova_lm(model, typ=3)
        print(anova_table)



# %%
print(os.getcwd())
PSIImaster = pd.read_excel('PSIImax-Master2-24.xlsx', engine='openpyxl')

# %%
PSIImaster['HeatMid'] = (PSIImaster['HeatUp'] + PSIImaster['HeatDown'])/2
PSIImaster['Heatrange'] = PSIImaster['HeatUp'] - PSIImaster['HeatDown']
PSIImaster.name = 'PSII Master'

PSIICrop = PSIImaster[PSIImaster['type'] == 'crop']
PSIICrop.name = 'PSII all Crop'
PSIITree = PSIImaster[PSIImaster['type'] == 'tree']
PSIITree.name = 'PSII all Tree'
PSIIGrass = PSIImaster[PSIImaster['type'] == 'grass-like']
PSIIGrass.name = 'PSII all Grass-like'
PSIIShrub = PSIImaster[PSIImaster['type'] == 'shrub']
PSIIShrub.name = 'PSII all shrub'

PSIIContr = PSIImaster[(PSIImaster['water status'] == 0) & (PSIImaster['nut status'] == 0)]
PSIIContr.name = 'PSII master Control'
# 7/28 Contr model values
#0     0.742302
#1    -9.348467
#2    12.000000
#3    48.374875
#4     7.357226
# use to create resid column
PSIIContr['resid'] = residualz(PSIIContr['HeatMid'], PSIIContr['phiPSIImax'],
                               [0.742302, -9.348467, 12.0, 48.374875, 7.357226])
# %%
# Creates the temp subdivisions
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
PSIICCrop = PSIIContr[PSIIContr['type'] == 'crop']
PSIICCrop.name = 'PSII control Crop'
PSIICTree = PSIIContr[PSIIContr['type'] == 'tree']
PSIICTree.name = 'PSII control Tree'
PSIICGrass = PSIIContr[PSIIContr['type'] == 'grass-like']
PSIICGrass.name = 'PSII control Grass-like'
PSIICShrub = PSIIContr[PSIIContr['type'] == 'shrub']
PSIICShrub.name = 'PSII control shrub'

print(np.unique(PSIIContr['PFT #']))
PFT1 = PSIIContr[PSIIContr['PFT #'] == 1] #needleleaf evergreen temp tree
PFT1.name = 'needleleaf evergreen temp tree'
PFT2 = PSIIContr[PSIIContr['PFT #'] == 2] #needleleaf evergreen boreal tree 24
PFT2.name = 'needleleaf evergreen boreal tree'
PFT3 = PSIIContr[PSIIContr['PFT #'] == 3] # needleleaf deciduous boreal tree
PFT3.name = 'needleleaf deciduous boreal tree'
PFT4 = PSIIContr[PSIIContr['PFT #'] == 4] #broadleaf evergreen trop tree 43
PFT4.name = 'broadleaf evergreen trop tree'
PFT5 = PSIIContr[PSIIContr['PFT #'] == 5] #broadleaf evergreen temp tree 3
PFT5.name = 'broadleaf evergreen temp tree'
PFT6 = PSIIContr[PSIIContr['PFT #'] == 6] #broadleaf deciduous trop tree 25
PFT6.name = 'broadleaf deciduous trop tree'
PFT7 = PSIIContr[PSIIContr['PFT #'] == 7] #broadleaf deciduous temp tree 30
PFT7.name = 'broadleaf deciduous temp tree'
PFT8 = PSIIContr[PSIIContr['PFT #'] == 8] # broadleaf deci boreal tree
PFT8.name = 'broadleaf deci boreal tree'
PFT9 = PSIIContr[PSIIContr['PFT #'] == 9] #broadleaf evergreen shrub
PFT9.name = 'broadleaf evergreen shrub'
PFT10 = PSIIContr[PSIIContr['PFT #'] == 10] # broadleaf deciduous temp shrub 48
PFT10.name = 'broadleaf deciduous temp shrub'
PFT11 = PSIIContr[PSIIContr['PFT #'] == 11] # broadleaf deciduous boreal shrub
PFT11.name = 'broadleaf deciduous boreal shrub'
PFT12 = PSIIContr[PSIIContr['PFT #'] == 12] # c3 arctic grass
PFT12.name = 'c3 arctic grass'
PFT13 = PSIIContr[PSIIContr['PFT #'] == 13] #c3 non-artic grass 3
PFT13.name = 'c3 non-arctic grass'
PFT14 = PSIIContr[PSIIContr['PFT #'] == 14] #c4 grass 7
PFT14.name = 'c4 grass'
PFT15 = PSIIContr[PSIIContr['PFT #'] == 15] #c3 crop 57
PFT15.name = 'c3 crop'
PFT16 = PSIIContr[PSIIContr['PFT #'] == 16] #c4 crop 
PFT16.name = 'c4 crop'
PFT17 = PSIIContr[PSIIContr['PFT #'] == 17] #temp corn 1 (add to PFT16)
PFT19 = PSIIContr[PSIIContr['PFT #'] == 19] #spring wheat 7 (add to PFT15)
PFT23 = PSIIContr[PSIIContr['PFT #'] == 23] #temp soybean (add to PFT15)
PFT39 = PSIIContr[PSIIContr['PFT #'] == 39] #coffee 14 Tree(add to PFT)
PFT41 = PSIIContr[PSIIContr['PFT #'] == 41] # cotton
PFT51 = PSIIContr[PSIIContr['PFT #'] == 51] # millet (add to PFT16)
PFT53 = PSIIContr[PSIIContr['PFT #'] == 53] #oilpalm 2 Tree(add to PFT)
PFT61 = PSIIContr[PSIIContr['PFT #'] == 61] #rice 13 (add to PFT15)
PFT63 = PSIIContr[PSIIContr['PFT #'] == 63] #sorghum 2 (add to PFT16)
PFT71 = PSIIContr[PSIIContr['PFT #'] == 71] #miscanthus (add to PFT14)
PFT73 = PSIIContr[PSIIContr['PFT #'] == 73] #switchgrass (add to PFT14)

FULLPFT14 = PFT14.append(PFT71).append(PFT73)
FULLPFT14.name = 'Aggrigate C4 grass PFT'
FULLPFT15 = PFT15.append(PFT19).append(PFT23).append(PFT61).append(PFT41)
FULLPFT15.name = 'Aggrigate C3 Crop PFT'
FULLPFT16 = PFT16.append(PFT17).append(PFT51).append(PFT63)
FULLPFT16.name = 'Aggrigate C4 Crop PFT'

# %%
dgrass = {'FvFm': PSIICGrass['phiPSIImax'], 'Temp': PSIICGrass['HeatMid'],
          'Age': PSIICGrass['Age (d)'], 'PFT': PSIICGrass['PFT #'],
          'Type': PSIICGrass['type'], 'Time': PSIICGrass['Time (h)']}
Dfgrass = pd.DataFrame(data=dgrass)
Time = PSIICGrass['Time (h)']
Temp = PSIICGrass['HeatMid']

# %%
aov = pg.anova(dv='FvFm', between=['Temp', 'Time', 'Type'], 
               data=Dfgrass, detailed=True)
print(aov)

aov1 = pg.mixed_anova(data=Dfgrass, dv='FvFm', between='Temp', within='Time',
                     subject='Subject', correction=False, effsize="np2")
print(aov1)
# %%
0.768
T = Plotter['Time (h)']
Y = Plotter['HeatMid']
Z = Plotter['phiPSIImax']
ax = plt.axes(projection='3d')
ax.view_init(0, 0) #adjusts the view (elevation and azimuth angles)
ax.scatter(T, Y, Z, c=T, cmap='viridis')
ax.set_xlabel('time (h)')
ax.set_ylabel('Temperature ' + u'\u2103')
ax.set_zlabel('Fv/Fm')

# %%
T = resid
Y = PFT13['HeatMid']
Z = PFT13['Time (h)']
ax = plt.axes(projection='3d')
ax.view_init(90, 0) #adjusts the view (elevation and azimuth angles)
ax.scatter(T, Y, Z, c=Z, cmap='viridis')
ax.set_xlabel('Residual')
ax.set_ylabel('Temperature ' + u'\u2103')
ax.set_zlabel('Time (h)')

dgrass13 = {'resid': resid, 'Temp': PFT13['HeatMid'],
          'Age': PFT13['Age (d)'], 'PFT': PFT13['PFT #'], 'Clim': PFT13['Climate'],
          'Type': PFT13['type'], 'Time': PFT13['Time (h)'], 'Perren': PFT13['Perrenial']}
Df13 = pd.DataFrame(data=dgrass13)
# %%
import statsmodels.api as sm
from statsmodels.formula.api import ols

dCont = {'FvFm': PSIIContr['phiPSIImax'], 'Temp': PSIIContr['temptemp'],
          'Age': PSIIContr['Age (d)'], 'PFT': PSIIContr['PFT #'],
          'Type': PSIIContr['type'], 'Time': PSIIContr['timetime']}
DfCont = pd.DataFrame(data=dCont)
model = ols('FvFm ~ C(Time) + C(Temp) + C(Type) + C(Time):C(Temp) + C(Time):C(Type)\
            + C(Temp):C(Type) + C(Time):C(Temp):C(Type)', data=DfCont).fit()
anova_table = sm.stats.anova_lm(model, typ=3)
anova_table
# %%
# this does not seem to be working, not giving expected output?
formula = 'FvFm ~ C(Time) + C(Temp) + C(PFT) + C(Time):C(Temp) + C(Time):C(PFT)\
            + C(Temp):C(PFT) + C(Time):C(Temp):C(PFT)'
model = ols(formula, DfCont).fit()
print(model.summary())
# %%
#plt.plot(abs(resid), PFT13['Perrenial'], 'o')
DGP0 = Df13[Df13['Perren'] == 0]
DGP1 = Df13[Df13['Perren'] == 1]
DGP2 = Df13[Df13['Perren'] == 2]
DGP3 = Df13[Df13['Perren'] == 3]
DGP4 = Df13[Df13['Perren'] == 4]
plt.boxplot([DGP0['resid'], DGP1['resid'], DGP3['resid'], DGP4['resid']],
             positions=[1, 1.5, 2.5, 3], widths=0.4, labels=[1, 2, 3, 4])
plt.title('Perrenial residue boxplot')

#  %%
plt.plot(DGP1['Temp'], DGP1['resid'], 'o')
plt.plot(DGP2['Temp'], DGP2['resid'], 'o')
plt.plot(DGP3['Temp'], DGP3['resid'], 'o')
plt.plot(DGP4['Temp'], DGP4['resid'], 'o')

# %%
Pfts = [PFT1, PFT2, PFT3, PFT4, PFT5, PFT6, PFT7, PFT8, PFT9, PFT10, PFT11, PFT12,
        PFT13, FULLPFT14, FULLPFT15, FULLPFT16]
Tipes = [PSIICCrop, PSIICGrass, PSIICShrub, PSIICTree]
Chooz = PFT13
for i in range(0,4):
    for j in range(0,5):
        print(i, j)
        if len(Chooz['phiPSIImax'][(Chooz['timetime'] == i) & (Chooz['temptemp'] == j)]) == 0:
            print('Value empty')
        else:
            print(np.mean(Chooz['phiPSIImax'][(Chooz['timetime'] == i) & (Chooz['temptemp'] == j)]))
MarginI = []
for i in range(0,4):
    print(i)
    MarginI.append(np.mean(PSIIContr['phiPSIImax'][(PSIIContr['Adjusted PFT'] == 13) & (PSIIContr['timetime'] == i)]))
MarginJ = []
for j in range(0,5):
    print(j)
    MarginJ.append(np.mean(Chooz['phiPSIImax'][(Chooz['temptemp'] == j)]))
MarginCross = np.zeros([4,5])
for i in range(0,4):
    for j in range(0,5):
        MarginCross[i,j] = np.mean(PSIIContr['phiPSIImax'][(PSIIContr['timetime'] == i) & (PSIIContr['temptemp'] == j)])

# %%
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

# %%
PSIIContr['Margin Call'] = PSIIContr['phiPSIImax']
for i in range(len(PSIIContr['phiPSIImax'])):
    l = PSIIContr['timetime'].iloc[i]
    j = PSIIContr['temptemp'].iloc[i]
    k = PSIIContr['Adjusted PFT'].iloc[i]

# %%
# 
TxPFT = np.zeros([16, 5])
for i in range(1,17):
    for j in range(0,5):
        TxPFT[i-1,j] = np.mean(PSIIContr['phiPSIImax'][(PSIIContr['Adjusted PFT'] == i) & (PSIIContr['temptemp'] == j)])

Tsolo = np.zeros(5)
for i in range(0, 5):
    Tsolo[i] = np.mean(PSIIContr['phiPSIImax'][(PSIIContr['temptemp'] == i)])

Pftsolo = np.zeros(16)
for i in range(1,17):
    Pftsolo[i-1] = np.mean(PSIIContr['phiPSIImax'][(PSIIContr['Adjusted PFT'] == i)])

mu = np.mean(PSIIContr['phiPSIImax'])

PSIIContr['ART 2 way'] = PSIIContr['phiPSIImax']
PSIIContr['ART x way'] = PSIIContr['phiPSIImax']
PSIIContr['ART y way'] = PSIIContr['phiPSIImax']
for i in range(len(PSIIContr['phiPSIImax'])):
    #print(i)
    l = int(PSIIContr['temptemp'].iloc[i])
    #print(l)
    k = int(PSIIContr['Adjusted PFT'].iloc[i])
    #print(k)
    PSIIContr['ART 2 way'].iloc[i] = (PSIIContr['phiPSIImax'].iloc[i] - Tsolo[l] - Pftsolo[k-1] + mu)
    PSIIContr['ART x way'].iloc[i] = (PSIIContr['phiPSIImax'].iloc[i] - TxPFT[k-1,l] + Tsolo[l] - mu)
    PSIIContr['ART y way'].iloc[i] = (PSIIContr['phiPSIImax'].iloc[i] - TxPFT[k-1,l] + Pftsolo[k-1] - mu)

PSIIranked = PSIIContr.sort_values(by='ART 2 way')
PSIIranked['number'] = PSIIranked['ART 2 way'].rank()

PSIIxranked = PSIIContr.sort_values(by='ART x way')
PSIIxranked['number'] = PSIIxranked['ART x way'].rank()

PSIIyranked = PSIIContr.sort_values(by='ART y way')
PSIIyranked['number'] = PSIIyranked['ART y way'].rank()

# %%
#for i in range(len(PSIIranked['phiPSIImax'])):
#    PSIIranked['number'].iloc[i] = len(PSIIranked['phiPSIImax']) - i
#    print(PSIIranked['number'].iloc[i])

import statsmodels.api as sm
from statsmodels.formula.api import ols

dCont = {'rank': PSIIyranked['number'], 'Temp': PSIIyranked['temptemp'], 'PFT': PSIIyranked['Adjusted PFT']}
DfCont = pd.DataFrame(data=dCont)
model = ols('rank ~ C(Temp) + C(PFT) + C(PFT):C(Temp) -1', data=DfCont).fit()
anova_table = sm.stats.anova_lm(model, typ=3)
anova_table
# %%
# performed on PFT13 to test for time contribution
Txtime13 = np.zeros([5, 4])
for i in range(0,5):
    for j in range(0,4):
        Txtime13[i,j] = np.mean(PFT13['phiPSIImax'][(PFT13['temptemp'] == i) & (PFT13['timetime'] == j)])

Tsolo13 = np.zeros(5)
for i in range(0, 5):
    Tsolo13[i] = np.mean(PFT13['phiPSIImax'][(PFT13['temptemp'] == i)])

Timesolo13 = np.zeros(4)
for i in range(0,4):
    Timesolo13[i] = np.mean(PFT13['phiPSIImax'][(PFT13['timetime'] == i)])

mu13 = np.mean(PFT13['phiPSIImax'])
#PFT13.dropna(axis=0, subset=['timetime'])

PFT13['ART 2 way'] = PFT13['phiPSIImax']
PFT13['ART x way'] = PFT13['phiPSIImax']
PFT13['ART y way'] = PFT13['phiPSIImax']
# The following line drops elements of PFT 13 based on the input of timetime, make sure to remake the 
# dataframe if this is not a relevant commponent of ongoing test
PFT13.dropna(subset=['timetime'], inplace=True)

for i in range(len(PFT13['phiPSIImax'])):
    print(i)
    l = int(PFT13['temptemp'].iloc[i])
    print(l)
    k = int(PFT13['timetime'].iloc[i])
    print(k)
    PFT13['ART 2 way'].iloc[i] = (PFT13['phiPSIImax'].iloc[i] - Tsolo13[l] - Timesolo13[k] + mu13)
    PFT13['ART x way'].iloc[i] = (PFT13['phiPSIImax'].iloc[i] - Txtime13[l,k] + Tsolo13[l] - mu13)
    PFT13['ART y way'].iloc[i] = (PFT13['phiPSIImax'].iloc[i] - Txtime13[l,k] + Timesolo13[k] - mu13)

PSIIxranked13 = PFT13.sort_values(by='ART x way')
PSIIxranked13['number'] = PSIIxranked13['ART x way'].rank()

PSIIyranked13 = PFT13.sort_values(by='ART y way')
PSIIyranked13['number'] = PSIIyranked13['ART y way'].rank()

PSIIranked13 = PFT13.sort_values(by='ART 2 way')
PSIIranked13['number'] = PSIIranked13['ART 2 way'].rank()

# %%
import statsmodels.api as sm
from statsmodels.formula.api import ols

dCont = {'rank': PSIIranked13['number'], 'Temp': PSIIranked13['temptemp'], 'Time': PSIIranked13['timetime']}
DfCont = pd.DataFrame(data=dCont)
model = ols('rank ~ C(Temp) + C(Time) + C(Time):C(Temp) -1', data=DfCont).fit()
anova_table = sm.stats.anova_lm(model, typ=3)
anova_table
# %%
PSII3ANOVA = PSIIContr
PSII3ANOVA.dropna(subset=['Adjusted PFT'])
PSII3ANOVA.dropna(subset=['temptemp'], inplace=True)
PSII3ANOVA.dropna(subset=['timetime'], inplace=True)
# %%
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
    print(x)
    i = int(PSII3ANOVA['temptemp'].iloc[x])
    print(i)
    j = int(PSII3ANOVA['timetime'].iloc[x])
    print(j)
    k = int(PSII3ANOVA['Adjusted PFT'].iloc[x])
    print(k)
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
import statsmodels.api as sm
from statsmodels.formula.api import ols
Pogg = PSIIranked3Way
dCont = {'rank': Pogg['number'], 'Temp': Pogg['temptemp'], 'Time': Pogg['timetime'], 'Species': Pogg['Adjusted PFT']}
DfCont = pd.DataFrame(data=dCont)
model = ols('rank ~ C(Temp) + C(Time) + C(Species) + C(Temp):C(Time) + C(Temp):C(Species) + C(Time):C(Species) + C(Temp):C(Time):C(Species) -1', data=DfCont).fit()
anova_table = sm.stats.anova_lm(model, typ=3)
anova_table
# %%
PFT13_3Way = PFT13
PFT13_3Way['ART T way'] = PFT13_3Way['phiPSIImax']
PFT13_3Way['ART t way'] = PFT13_3Way['phiPSIImax']
PFT13_3Way['ART S way'] = PFT13_3Way['phiPSIImax']
PFT13_3Way['ART Tt way'] = PFT13_3Way['phiPSIImax']
PFT13_3Way['ART TS way'] = PFT13_3Way['phiPSIImax']
PFT13_3Way['ART tS way'] = PFT13_3Way['phiPSIImax']
PFT13_3Way['ART TtS way'] = PFT13_3Way['phiPSIImax']
# The following line drops elements of PFT 13 based on the input of timetime, make sure to remake the 
# dataframe if this is not a relevant commponent of ongoing test
PFT13_3Way.dropna(subset=['timetime'], inplace=True)

for x in range(len(PFT13_3Way['phiPSIImax'])):
    print(x)
    i = int(PFT13_3Way['temptemp'].iloc[x])
    print(i)
    j = int(PFT13_3Way['timetime'].iloc[x])
    print(j)
    k = int(PFT13_3Way['Adjusted PFT'].iloc[x])
    print(k)
    PFT13_3Way['ART T way'].iloc[x] = (PFT13_3Way['phiPSIImax'].iloc[x] - np.mean(PFT13_3Way['phiPSIImax'][(PFT13_3Way['temptemp'] == i) & (PFT13_3Way['timetime'] == j) & (PFT13_3Way['Adjusted PFT'] == k)]) + Tsolo[i] - mu)
    PFT13_3Way['ART t way'].iloc[x] = (PFT13_3Way['phiPSIImax'].iloc[x] - np.mean(PFT13_3Way['phiPSIImax'][(PFT13_3Way['temptemp'] == i) & (PFT13_3Way['timetime'] == j) & (PFT13_3Way['Adjusted PFT'] == k)]) + Timesolo[j] - mu)
    PFT13_3Way['ART S way'].iloc[x] = (PFT13_3Way['phiPSIImax'].iloc[x] - np.mean(PFT13_3Way['phiPSIImax'][(PFT13_3Way['temptemp'] == i) & (PFT13_3Way['timetime'] == j) & (PFT13_3Way['Adjusted PFT'] == k)]) + Speciessolo[k-1] - mu)
    PFT13_3Way['ART Tt way'].iloc[x] = (PFT13_3Way['phiPSIImax'].iloc[x] - np.mean(PFT13_3Way['phiPSIImax'][(PFT13_3Way['temptemp'] == i) & (PFT13_3Way['timetime'] == j) & (PFT13_3Way['Adjusted PFT'] == k)]) + Txtime[i,j] - Tsolo[i] - Timesolo[j] + mu)
    PFT13_3Way['ART TS way'].iloc[x] = (PFT13_3Way['phiPSIImax'].iloc[x] - np.mean(PFT13_3Way['phiPSIImax'][(PFT13_3Way['temptemp'] == i) & (PFT13_3Way['timetime'] == j) & (PFT13_3Way['Adjusted PFT'] == k)]) + TxSpec[i,k-1] - Tsolo[i] - Speciessolo[k-1] + mu)
    PFT13_3Way['ART tS way'].iloc[x] = (PFT13_3Way['phiPSIImax'].iloc[x] - np.mean(PFT13_3Way['phiPSIImax'][(PFT13_3Way['temptemp'] == i) & (PFT13_3Way['timetime'] == j) & (PFT13_3Way['Adjusted PFT'] == k)]) + timexSpec[j,k-1] - Timesolo[j] - Speciessolo[k-1] + mu)
    PFT13_3Way['ART TtS way'].iloc[x] = (PFT13_3Way['phiPSIImax'].iloc[x] - Txtime[i,j] - TxSpec[i,k-1] - timexSpec[j,k-1] + Tsolo[i] + Timesolo[j] + Speciessolo[k-1] - mu)


PSIITranked3Way = PFT13_3Way.sort_values(by='ART T way')
PSIITranked3Way['number'] = PSIITranked3Way['ART T way'].rank()
PSIItranked3Way = PFT13_3Way.sort_values(by='ART t way')
PSIItranked3Way['number'] = PSIItranked3Way['ART t way'].rank()
PSIISranked3Way = PFT13_3Way.sort_values(by='ART S way')
PSIISranked3Way['number'] = PSIISranked3Way['ART S way'].rank()
PSIITtranked3Way = PFT13_3Way.sort_values(by='ART Tt way')
PSIITtranked3Way['number'] = PSIITtranked3Way['ART Tt way'].rank()
PSIITSranked3Way = PFT13_3Way.sort_values(by='ART TS way')
PSIITSranked3Way['number'] = PSIITSranked3Way['ART TS way'].rank()
PSIItSranked3Way = PFT13_3Way.sort_values(by='ART tS way')
PSIItSranked3Way['number'] = PSIItSranked3Way['ART tS way'].rank()
PSIIranked3Way = PFT13_3Way.sort_values(by='ART TtS way')
PSIIranked3Way['number'] = PSIIranked3Way['ART TtS way'].rank()

# %%
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py
category_names = ['Temp', 'Time', 'PFT', 'Temp x time', 'Temp x PFT',
                  'time x PFT', '3-way interaction', 'residuals']
results = {
    '2-way All': [0.156518, 0, 0.04921, 0, 0.218508, 0, 0, 0.575767],
    '2-way PFT 13': [0.332396, 0.0294123, 0, 0.1993399, 0, 0, 0, 0.4388515],
    '3-way All': [0.0456735, 0.0025724, 0.0145984, 0.0615572, 0.1179451, 0.10674004, 0.2771663, 0.37374696]
}


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        #ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


survey(results, category_names)
plt.show()
# %%
# https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py
fig, axs = plt.subplots(1,1, figsize=[10,10])
#axs.pie([15.6518, 4.921, 21.8508, 57.5767], labels=['Temp', 'PFT', 'Temp x PFT', 'residuals'],
#             autopct='%1.0f%%', pctdistance=.9, labeldistance=1.1, radius=1, textprops={'fontsize': 15})
#axs.set_title('2-way All data', fontsize=15)
axs.pie([33.2396, 2.94123, 19.93399, 43.88515], labels=['Temp', 'Time', 'Temp x time', 'residuals'],
             autopct='%1.0f%%', pctdistance=.9, labeldistance=1.1, radius=1, textprops={'fontsize': 15})
axs.set_title('2-way PFT 13', fontsize=15)
#axs.pie([4.56735, 0.25724, 1.45984, 6.15572, 11.79451, 10.674004, 27.71663, 37.374696],
#             labels=['Temp', 'Time', 'PFT', 'Temp x time', 'Temp x PFT', 'time x PFT', '3-way interaction', 'residuals'],
#             autopct='%1.0f%%', pctdistance=.9, labeldistance=1.1, radius=1, textprops={'fontsize': 15}, startangle=-12)
#axs.set_title('3-way All data', fontsize=15)
# %%
