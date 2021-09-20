# PSII Master Attempt #1 (2-24-21)
# Produced with the intention to work from a master list of a
# spreadsheet containing all variables with dynamic plotting
# possible.
# Done by Patrick Neri

# %%
# Currently working out of hastools environment
from lmfit.model import save_model
from matplotlib import colors
import numpy as np
from numpy.core.function_base import linspace
import pandas as pd
import matplotlib.pyplot as plt
import math
# from scipy.optimize import curve_fit
from lmfit import Model
from lmfit.models import RectangleModel, StepModel
from sklearn.metrics import r2_score
import scipy.stats as stats

# %%
# all current relevent functions used

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

def Quadratic(x, amp, a1, a2):
    """1-d quadratic: quadratic(x, amp, a1, a2)"""
    return (amp * ((x - a1)**2) + a2)

def Trial_Reaction_Equ(x, tnaught, n, EsubT):
    """Equation described by prof Song and Lianghou"""
    return (((tnaught/x)**n)*((tnaught/x)**0.5))*np.exp(EsubT*((1/tnaught) - (1/x)))

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

def boxplot_outliers(dataset):
        "This is the description"
        QuantFvFm = dataset['phiPSIImax'].quantile([0.25, 0.5, 0.75])
        outliershigh = dataset[dataset['phiPSIImax'] > (QuantFvFm.iloc[2] + 1.5*(QuantFvFm.iloc[2]-QuantFvFm.iloc[0]))] 
        outlierslow = dataset[dataset['phiPSIImax'] < (QuantFvFm.iloc[0] - 1.5*(QuantFvFm.iloc[2]-QuantFvFm.iloc[0]))]
        alloutliers = outliershigh.append(outlierslow)
        return alloutliers

def unique(list1):
    # intilize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    print(len(unique_list))
    for x in unique_list:
        print(x)
    return unique_list    


# Goal is to add a function that can isolate variable as well as plant type
# and produce a df (list?)


def get_zoomedDiff(dataset1, dataset2):

        zoomedDiff = np.zeros(len(dataset))
        for i in range(0, 500):
                zoomedDiff[i] = ContRun[i] - ContSRun[i]
        
def FullPlot_slopped(dataset, ModelFit):
    """ Showing a plot with 2 slopes and max, plus 2 sigma on each side"""
    runtri = np.linspace(-20, 65, 500)
    TITle = input('Please type the title of the plot')
    A, m1, s1, m2, s2 = ModelFit[:5] 
    front = (runtri - m1)/s1
    UpperRun = []
    for i in range(len(front)):
        UpperRun.append(math.erf(front[i]))
    back = -(runtri - m2)/s2
    LowerRun = []
    for i in range(len(back)):
        LowerRun.append(math.erf(back[i]))
    ContRun = []
    for i in range(len(UpperRun)):
       ContRun.append((A/2)* (UpperRun[i] + LowerRun[i]))
    
    y = np.arange(0.0, 1.2, 0.01)
    x1 = ModelFit[1]
    x2 = ModelFit[1] + ModelFit[2]
    x3 = ModelFit[1] + 2 * ModelFit[2]
    x4 = ModelFit[3]
    x5 = ModelFit[3] - ModelFit[4]
    x6 = ModelFit[3] - 2 * ModelFit[4]
    line = ((1/(2 * ModelFit[2])) * (runtri - ModelFit[1]) + (ModelFit[0] / 2))
    line2 = (-(1/(2 * ModelFit[4])) * (runtri - ModelFit[3]) + (ModelFit[0] / 2))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill_betweenx(y, x1, x2, alpha=0.1, color='g')
    ax.fill_betweenx(y, x2, x3, alpha=0.1, color='k')
    ax.fill_betweenx(y, x5, x4, alpha=0.1, color='g')
    ax.fill_betweenx(y, x6, x5, alpha=0.1, color='k')
    ax.plot(dataset['HeatMid'], dataset['phiPSIImax'], 'o', alpha=0.2)
    ax.vlines(ModelFit[5], 0, 1.2)
    ax.plot(runtri, ContRun, label=dataset.name)
    ax.plot(runtri, line, '--g')
    ax.plot(runtri, line2, '--r')
    ax.hlines([A, (A / 2)], -17, 62, colors=['b', 'c'])
    ax.set_xlabel('Temperature ' + u'\u2103')
    ax.set_ylabel('Maximum Quantum Efficiency of PSII [0-1]')
    ax.set_xlim([-17, 62])
    ax.set_ylim([0, 1])
    ax.set_title(TITle)
    ax.grid(True)


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

def minmax(val_list):
    min_val = min(val_list)
    max_val = max(val_list)

    return (min_val, max_val)


def PFT_numbers(PFTdataset):
        print('# of data points is ' + str(len(PFTdataset.index)))
        print('# of papers is ' + str(len(np.unique(PFTdataset['paper']))))
        print('Range of data is ' + str(minmax(PFTdataset['HeatMid'])))
        plt.plot(PFTdataset['HeatMid'], PFTdataset['phiPSIImax'], 'o')
        plt.grid(True)
        plt.ylim([0,1])
        plt.show()


# %%
# Pull in all relevent documents here
print(os.getcwd())
PSIImaster = pd.read_excel('PSIImax-Master2-24.xlsx')
#ModelFits = pd.read_excel('PSIImodelfits6-14.xlsx')
# Adjustments to Clade, Order, Family columns
PSIImaster['Clade A'] = PSIImaster['Clade A'].astype(str)
PSIImaster['Clade B'] = PSIImaster['Clade B'].astype(str)
PSIImaster['Order'] = PSIImaster['Order'].astype(str)
PSIImaster['Family'] = PSIImaster['Family'].astype(str)
# %%
# Divide up into subplot datasets
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
PSIICCrop = PSIIContr[PSIIContr['type'] == 'crop']
PSIICCrop.name = 'PSII control Crop'
PSIICTree = PSIIContr[PSIIContr['type'] == 'tree']
PSIICTree.name = 'PSII control Tree'
PSIICGrass = PSIIContr[PSIIContr['type'] == 'grass-like']
PSIICGrass.name = 'PSII control Grass-like'
PSIICShrub = PSIIContr[PSIIContr['type'] == 'shrub']
PSIICShrub.name = 'PSII control shrub'

# I don't think we want these anymore... Useful to pull papers that specifically vary temp
PSIIHeat = PSIImaster[PSIImaster['temp status'] == 1] #Might need to change this...
PSIIHeatACrop = PSIIHeat[PSIIHeat['type'] == 'crop']
PSIIHeatATree = PSIIHeat[PSIIHeat['type'] == 'tree']
PSIIHeatAGrass = PSIIHeat[PSIIHeat['type'] == 'grass-like']

PSIIHeatContr = PSIIHeat[(PSIIHeat['water status'] == 0) & (PSIIHeat['nut status'] == 0)]
PSIIHeatCCrop = PSIIHeatContr[PSIIHeatContr['type'] == 'crop']
PSIIHeatCTree = PSIIHeatContr[PSIIHeatContr['type'] == 'tree']
PSIIHeatCGrass = PSIIHeatContr[PSIIHeatContr['type'] == 'grass-like']

# %%
# Cut content, see RemovedfromMasterAttempt2-24-2.py
# %%
# Here is all point plots, to check the situation
# Cut content, see RemovedfromMasterAttempt2-24-2.py
# %%
# Here is a proof of concept for Rect Model and its r-squared
# Cut content, see RemovedfromMasterAttempt2-24-2.py
# %%
#A, m1, s1, m2, s2, M = out.
def get_setdata_plot(dataset):
        """ Pick a chosen dataset that has a HeatMid column and phiPSIImax column"""
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
        pars['center1'].set(value=-6, min=-12, max=7)
        pars['center2'].set(value=46, min=35, max=57)
        pars['sigma1'].set(value=7, min=1, max=12)
        pars['sigma2'].set(value=5, min=1, max=12)
        out = mod.fit(y, pars, x=x)
        ModelChoice = 'Rect' #input('Chose between [Quad, Rect] or [Both] models:')
        if ModelChoice == 'Rect':
                print('You have chosen Rectangle model for the ', dataset.name, 'dataset.')
                print(out.fit_report())
                ps = get_Mod_paramsValues(out)
                print(ps.val[:])
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
                print('R-squared : ', r2_score(y, Ayy))

                # Here the residual of the dataset is calculated
                resid = (y - Ayy)
                # Below is a histogram of the residual values
                fig = plt.figure(figsize=(6, 3))
                gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                                      wspace=0.05, hspace=0.05)
                ax = fig.add_subplot(gs[1, 0])
                ax.plot(x, resid, 'o')
                ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
                ax_histy.hist(resid, bins=40, orientation='horizontal')
                ax_histy.tick_params(axis="y", labelleft=False)
                #ax.set_title('Residuals of ' + dataset.name)
                ax.grid(True)
                #plt.savefig(dataset.name + 'residue5-28-21.JPG')
                # Below is plot, r-squared functionality achieved
                plt.figure(2, figsize=(5,4))
                
                plt.plot(x, y, 'k+', alpha=0.6)
                plt.plot(x, out.best_fit, 'r-', label='best fit')
                dely = out.eval_uncertainty(sigma=1)
                plt.fill_between(x, out.best_fit-dely, out.best_fit+dely, color="#ABABAB",
                 label='1-$\sigma$ \nuncertainty band')
                plt.legend(loc='best')
                plt.ylim([0, 1])
                plt.xlim([-17, 62])
                plt.ylabel('Maximum Quantum \nEfficiency of PSII', fontsize=14)
                plt.xlabel('Temperature ' + u'\u2103', fontsize=14)
                #plt.title('Rectangular (erf) model fit of ' + dataset.name +' with uncertainty')
                plt.grid(True)
                plt.annotate('$\mathregular{R^{2}}$ - ' + str(round(r2_score(y, Ayy), 2)) + '\nN = ' + str(len(y.index)), xy=(1,1),
                             xycoords='axes fraction', xytext=(-10, -10), textcoords='offset pixels',
                             horizontalalignment='right',
                             verticalalignment='top')
                plt.show()
                #plt.savefig(dataset.name + 'model5-28-21.JPG')
                # Below is the attempt at formatting a boxplot figure (can be optimized)
                TrialBox = dataset.sort_values(by='HeatMid')
                
                Box0 = TrialBox[(TrialBox['HeatMid'] > -17) & (TrialBox['HeatMid'] < -7)]
                Box1 = TrialBox[(TrialBox['HeatMid'] > -7) & (TrialBox['HeatMid'] < 3)]
                Box2 = TrialBox[(TrialBox['HeatMid'] > 3) & (TrialBox['HeatMid'] < 13)]
                Box3 = TrialBox[(TrialBox['HeatMid'] > 13) & (TrialBox['HeatMid'] < 23)]
                Box4 = TrialBox[(TrialBox['HeatMid'] > 23) & (TrialBox['HeatMid'] < 33)]
                Box5 = TrialBox[(TrialBox['HeatMid'] > 33) & (TrialBox['HeatMid'] < 43)]
                Box6 = TrialBox[(TrialBox['HeatMid'] > 43) & (TrialBox['HeatMid'] < 53)]
                Box7 = TrialBox[(TrialBox['HeatMid'] > 53) & (TrialBox['HeatMid'] < 63)]
                plt.figure(3)
                plt.boxplot([Box0['phiPSIImax'], Box1['phiPSIImax'], Box2['phiPSIImax'], Box3['phiPSIImax'],
                             Box4['phiPSIImax'], Box5['phiPSIImax'], Box6['phiPSIImax'], Box7['phiPSIImax']],
                            positions=(-12, -2, 8, 18, 28, 38, 48, 58), widths=10)
                plt.plot(x, y, 'o', alpha=0.3)
                #plt.title(dataset.name + ' Boxplot (width 10) with data')
                plt.ylim([0, 1])
                plt.grid(True)
                #plt.savefig(dataset.name + 'Box5-28-21.JPG')  
                FullPlot_slopped(dataset, [ps.val[0], ps.val[1], ps.val[2],
                                           ps.val[3], ps.val[4],
                                           ((ps.val[1] + ps.val[3])*0.5)])

# %%
# To do: Use the split, str, and float functions to isolate (done)
# Get r-2 values into plot
# consider doing in frame subplot comparison with all data (control or all?) model plots.
# produce largescale all 8 plots in one graphical output (End goal)
# potentially make a function that allows division of master plot in initiating step
#       So that you dont have to enter a full dataframe name every time?
# 
# %%
TrialBox = PSIIContr.sort_values(by='HeatMid')
Box0 = TrialBox[(TrialBox['HeatMid'] > -17) & (TrialBox['HeatMid'] < -12)] #-7
Box1 = TrialBox[(TrialBox['HeatMid'] > -7) & (TrialBox['HeatMid'] < -2)] #3
Box2 = TrialBox[(TrialBox['HeatMid'] > 3) & (TrialBox['HeatMid'] < 8)] #13
Box3 = TrialBox[(TrialBox['HeatMid'] > 13) & (TrialBox['HeatMid'] < 18)] #23
Box4 = TrialBox[(TrialBox['HeatMid'] > 23) & (TrialBox['HeatMid'] < 28)] #33
Box5 = TrialBox[(TrialBox['HeatMid'] > 33) & (TrialBox['HeatMid'] < 38)] #43
Box6 = TrialBox[(TrialBox['HeatMid'] > 43) & (TrialBox['HeatMid'] < 48)] #53
Box7 = TrialBox[(TrialBox['HeatMid'] > 53) & (TrialBox['HeatMid'] < 58)] #63

Box0_5 = TrialBox[(TrialBox['HeatMid'] > -12) & (TrialBox['HeatMid'] < -7)]
Box1_5 = TrialBox[(TrialBox['HeatMid'] > -2) & (TrialBox['HeatMid'] < 3)]
Box2_5 = TrialBox[(TrialBox['HeatMid'] > 8) & (TrialBox['HeatMid'] < 13)]
Box3_5 = TrialBox[(TrialBox['HeatMid'] > 18) & (TrialBox['HeatMid'] < 23)]
Box4_5 = TrialBox[(TrialBox['HeatMid'] > 28) & (TrialBox['HeatMid'] < 33)]
Box5_5 = TrialBox[(TrialBox['HeatMid'] > 38) & (TrialBox['HeatMid'] < 43)]
Box6_5 = TrialBox[(TrialBox['HeatMid'] > 48) & (TrialBox['HeatMid'] < 53)]
Box7_5 = TrialBox[(TrialBox['HeatMid'] > 58) & (TrialBox['HeatMid'] < 63)]
plt.plot(TrialBox['HeatMid'], TrialBox['phiPSIImax'], 'o', alpha=0.3)
plt.boxplot([Box0['phiPSIImax'], Box1['phiPSIImax'], Box2['phiPSIImax'], Box3['phiPSIImax'],
             Box4['phiPSIImax'], Box5['phiPSIImax'], Box6['phiPSIImax'], Box7['phiPSIImax'],
             Box0_5['phiPSIImax'], Box1_5['phiPSIImax'], Box2_5['phiPSIImax'], Box3_5['phiPSIImax'],
             Box4_5['phiPSIImax'], Box5_5['phiPSIImax'], Box6_5['phiPSIImax'], Box7_5['phiPSIImax']],
             positions=(-14.5, -4.5, 5.5, 15.5, 25.5, 35.5, 45.5, 55.5,
                       -9.5, 0.5, 10.5, 20.5, 30.5, 40.5, 50.5, 60.5), 
             labels=(0, 1, 2, 3, 4, 5 ,6, 7, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5), widths=5)
plt.rcParams["figure.figsize"] = (12,10)
plt.xticks(range(-32,68, 5))
plt.yticks(np.linspace(0,1,6))
plt.grid()
plt.ylim([0, 1])
plt.title('PSIIContr Boxplot (width 5) with data')
# %%
# determine outliers (I hate making functions)
Boxlist = [Box0, Box1, Box2, Box3, Box4, Box5, Box6, Box7, Box0_5, Box1_5, Box2_5, Box3_5,
           Box4_5, Box5_5, Box6_5, Box7_5]
OutIndex = []
for i in range(1,16):
        Boxi_ = boxplot_outliers(Boxlist[i])
        print(Boxi_.index)
        OutIndex.append(Boxi_.index)

# %%
# updated (as of 7/14 noon)
PSIIContr_sans = PSIIContr.drop([673,697,1421,362,361,1459,1457,1465,288,1199,124,1145,1200,1201,488,132,241,136,1147,1182,4,6,238,
                                 178,177,179,170,168,126,90,1167,1158,39,53,57,92,355,356,282,286,283,1387,1482,1490,468,237,472,1204,
                                 1089,1109,1108,1107,1106,1105,1104,1103,127,91,1159,1471,348,347], axis=0)
PSIIContr_sans.name = 'PSII Control sans Boxplot outliers'
PSIICCrop_sans = PSIICCrop.drop([485,488,2111,251,1575,6,4,164,170,168,126,2113,669,668,667,20,100,122,22,486,483,
                                 484,487,751], axis=0)
PSIICCrop_sans.name = 'PSII control Crop sans Boxplot outliers'
PSIICGrass_sans = PSIICGrass.drop([705,681,689,1971,1966,1965,1947,1176,1182,2107,2108,177,2020,2019,2018,2017,
                                   2028,2133,2129,2105,384,1995,1996,1109,1645,2006], axis=0)
PSIICGrass_sans.name = 'PSII control Grass_like sans Boxplot outliers'
PSIICTree_sans = PSIICTree.drop([1594,1600,288,1777,1772,1711,1713,1717,1556,1793,1778,1557,241,238,179,178,39,53,
                                 1473,355,356,1387,1389,1630,1893,1870,1428,1482,1490,1599,1622,1930,1750,1775,237,
                                 1815,1810,1558,1897,1901,1740,1741,1471,347], axis=0)
PSIICTree_sans.name = 'PSII control Tree sans Boxplot outliers'
PSIICShrub_sans = PSIICShrub.drop([87,1199,90,57,1148,91,1159,2200], axis=0)
PSIICShrub_sans.name = 'PSII control Shrub sans Boxplot outliers'
# %%
# As of 4/15/2021
print(len(np.unique(PSIImaster['paper']))) # 267 points
print(len(np.unique(PSIIContr['paper']))) # 189 points
print(len(np.unique(PSIICrop['paper']))) # 123 points
print(len(np.unique(PSIICCrop['paper']))) # 81 points
print(len(np.unique(PSIIGrass['paper']))) # 60 points
print(len(np.unique(PSIICGrass['paper']))) # 52 points
print(len(np.unique(PSIITree['paper']))) # 84 points
print(len(np.unique(PSIICTree['paper']))) # 56 points
print(len(np.unique(PSIIShrub['paper'])))
print(len(np.unique(PSIICShrub['paper'])))
print(len(np.unique(PSIIContr_sans['paper'])))
print(len(np.unique(PSIICCrop_sans['paper'])))
print(len(np.unique(PSIICTree_sans['paper'])))
print(len(np.unique(PSIICGrass_sans['paper'])))
print(len(np.unique(PSIICShrub_sans['paper'])))

# %%
# Here was a test that turned into the box_plotoutliers function
# %%
# all current PFT in excel plot
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
PFT11 = PSIIContr[PSIIContr['PFT #'] == 11]
PFT11.name = 'broadleaf deciduous boreal shrub' # broadleaf deciduous boreal shrub
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
PFT39.name = 'coffee'
PFT41 = PSIIContr[PSIIContr['PFT #'] == 41] # cotton
PFT51 = PSIIContr[PSIIContr['PFT #'] == 51] # millet (add to PFT16)
PFT53 = PSIIContr[PSIIContr['PFT #'] == 53] #oilpalm 2 Tree(add to PFT)
PFT61 = PSIIContr[PSIIContr['PFT #'] == 61] #rice 13 (add to PFT15)
PFT63 = PSIIContr[PSIIContr['PFT #'] == 63] #sorghum 2 (add to PFT16)
PFT71 = PSIIContr[PSIIContr['PFT #'] == 71] #miscanthus (add to PFT14)
PFT73 = PSIIContr[PSIIContr['PFT #'] == 73] #switchgrass (add to PFT14)

# For Summary purposes until more data is gained
FULLPFT4 = PFT4.append(PFT39).append(PFT53)
FULLPFT4.name = 'Aggrigate Broadleaf evergreen trop tree'
FULLPFT14 = PFT14.append(PFT71).append(PFT73)
FULLPFT14.name = 'Aggrigate C4 grass PFT'
FULLPFT15 = PFT15.append(PFT19).append(PFT23).append(PFT61).append(PFT41)
FULLPFT15.name = 'Aggrigate C3 Crop PFT'
FULLPFT16 = PFT16.append(PFT17).append(PFT51).append(PFT63)
FULLPFT16.name = 'Aggrigate C4 Crop PFT'

# Adjustment for time issues specific to PFT 13
TimePFT13 = PFT13[PFT13['Time (h)'] < 49]
TimePFT13.name = 'PFT13 t < 49 h'

# %%
for i in range(0,16):
        if np.unique(PSIIContr['PFT #'])[i] != 'nan':
                PFTN = PSIIContr[PSIIContr['PFT #'] == np.unique(PSIIContr['PFT #'])[i]]
                print('Info for PFT # ' + str(np.unique(PSIIContr['PFT #'])[i]))
                PFT_numbers(PFTN)
        else:
                print('Completed')

print('Info for PFT # 13 Time < 54 h')
PFT_numbers(TimePFT13)
print('Info for PFT # 14 FULL' )
PFT_numbers(FULLPFT14)
print('Info for PFT # 15 FULL')
PFT_numbers(FULLPFT15)
print('Inof for PFT # 16 FULL')
PFT_numbers(FULLPFT16)
print('Inof for PFT # 4 FULL')
PFT_numbers(FULLPFT4)

# %%
# tree PFT examination
plt.plot(PFT1['HeatMid'], PFT1['phiPSIImax'], 'o', label='1 temp')
plt.plot(PFT2['HeatMid'], PFT2['phiPSIImax'], 'o', label='2 bor')
plt.plot(PFT4['HeatMid'], PFT4['phiPSIImax'], 'o', label='4 trop')
plt.plot(PFT5['HeatMid'], PFT5['phiPSIImax'], 'o', label='5 temp')
plt.plot(PFT6['HeatMid'], PFT6['phiPSIImax'], 'o', label='6 trop')
plt.plot(PFT7['HeatMid'], PFT7['phiPSIImax'], 'o', label='7 temp')
#optional
plt.plot(PFT39['HeatMid'], PFT39['phiPSIImax'], 'o', label='39 coffee')
plt.legend()
plt.title('Tree PFT breakdown')
plt.grid(True)
plt.ylim([0,1])
# %%
# crop PFT examination?
plt.plot(PFT15['HeatMid'], PFT15['phiPSIImax'], 'o', label='15 crop')
plt.plot(PFT17['HeatMid'], PFT17['phiPSIImax'], 'o', label='17 corn')
plt.plot(PFT19['HeatMid'], PFT19['phiPSIImax'], 'o', label='19 wheat')
plt.plot(PFT61['HeatMid'], PFT61['phiPSIImax'], 'o', label='61 rice')
plt.plot(PFT63['HeatMid'], PFT63['phiPSIImax'], 'o', label='63 sorghum')
plt.plot(PFT53['HeatMid'], PFT53['phiPSIImax'], 'o', label='53 oilpalm')
plt.plot(PFT39['HeatMid'], PFT39['phiPSIImax'], 'o', label='39 coffee')
plt.plot(PFT23['HeatMid'], PFT23['phiPSIImax'], 'o', label='23 soybean')
#optional
plt.plot(PFT51['HeatMid'], PFT51['phiPSIImax'], 'o', label='51 millet')
plt.legend()
plt.title('Crop PFT breakdown')
plt.grid(True)
plt.ylim([0,1])
# %%
# grass-like PFT examination?
plt.plot(PFT13['HeatMid'], PFT13['phiPSIImax'], 'o', label='13 non-arctic grass')
plt.plot(PFT14['HeatMid'], PFT14['phiPSIImax'], 'o', label='14 c4 grass')
plt.plot(PFT73['HeatMid'], PFT73['phiPSIImax'], 'o', label='73 switchgrass')
plt.plot(PFT71['HeatMid'], PFT71['phiPSIImax'], 'o', label='71 miscanthus')
#optional
plt.plot(PFT51['HeatMid'], PFT51['phiPSIImax'], 'o', label='51 millet')
plt.legend()
plt.title('Grass-like PFT breakdown')
plt.grid(True)
plt.ylim([0,1])
#%%
plt.plot(PFT9['HeatMid'], PFT9['phiPSIImax'], 'o', label='9 ever')
plt.plot(PFT10['HeatMid'], PFT10['phiPSIImax'], 'o', label='10 deci temp')
plt.plot(PFT11['HeatMid'], PFT11['phiPSIImax'], 'o', label='11 deci bor')
plt.legend()
plt.title('Grass-like PFT breakdown')
plt.grid(True)
plt.ylim([0,1])

# %%
#attempt to make an easier step for function variable calls in FullPlot_Slopped
#removed
# %%
#attached to above call removed
# %%
#removed, example plots and comparisons of model before and after removal of outliers
# %%
# moved functions above
# %%
# Below is a 3d plot for over time
from mpl_toolkits import mplot3d
# %%
TwoDayPlot = PSIIContr[(PSIIContr['Time (h)'] < 1000)]
Plotter = PSIIContr
T = Plotter['Age (d)']
Y = Plotter['HeatMid']
Z = Plotter['phiPSIImax']
ax = plt.axes(projection='3d')
ax.view_init(90, 0) #adjusts the view (elevation and azimuth angles)
ax.scatter(T, Y, Z, c=T, cmap='viridis')
ax.set_xlabel('Age (d)')
ax.set_ylabel('Temperature ' + u'\u2103')
ax.set_zlabel('Fv/Fm')
# %%
# Below is a 2D color plot over time
# very rough, will need a lot of data to do averaging on a grid I think
ax = plt.axes()
ax.scatter(T, Y, c=Z, cmap='viridis')
ax.set_xlim([0,12])
# %%
Treeplotset = PSIICTree[PSIICTree['HeatMid'] > -17]
Treeplotset.name = 'Adjusted Tree 1'
# %%
# attempted resource
# http://pytolearn.csd.auth.gr/d1-hyptest/12/ptl_anova2.html

# another helpful framework
# https://www.statology.org/two-way-anova-python/

# %%
dgrass = {'Fv/Fm': PSIICGrass['phiPSIImax'], 'Temp': PSIICGrass['HeatMid'],
          'Age': PSIICGrass['Age (d)'], 'PFT': PSIICGrass['PFT #'],
          'Type': PSIICGrass['type'], 'Time': PSIICGrass['Time (h)']}
Dfgrass = pd.DataFrame(data=dgrass)


# %%
def get_spread_plot(dataset):
        """ Pick a chosen dataset that has a HeatMid column and phiPSIImax column"""
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
        out = mod.fit(y, pars, x=x)
        ps = get_Mod_paramsValues(out)
        print(ps.val[:])
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

# %%
from lmfit import Minimizer, Parameters, report_fit, minimize

x = PSIICTree['HeatMid']
data = PSIICTree['phiPSIImax']

def RectMinAttempt(params, x, data=None):
        amp = params['amp']
        cen1 = params['center1']
        sig1 = params['sigma1']
        cen2 = params['center2']
        sig2 = params['sigma2']

        Aa1 = (x - cen1)/sig1
        Aaa1 = []
        for i in range(len(Aa1)):
                Aaa1.append(math.erf(Aa1[i]))
        Aa2 = -(x - cen2)/sig2
        Aaa2 = []
        for i in range(len(Aa2)):
                Aaa2.append(math.erf(Aa2[i]))
        Ayy = []
        for i in range(len(Aaa1)):
               Ayy.append((amp/2)* (Aaa1[i] + Aaa2[i]))
        Ayyy = pd.DataFrame(data=Ayy)
        model = Ayyy
        if data is None:
                return model
        return model - data

params = Parameters()
params.add('amp', value=0.8, min=0.6, max=0.83)
params.add('center1', value=-6, min=-12, max=7)
params.add('sigma1', value=7, min=1, max=12)
params.add('center2', value=46, min=35, max=57)
params.add('sigma2', value=5, min=1, max=12)

out = minimize(RectMinAttempt, params, args=(x,), kws={'data': data})
fit = RectMinAttempt(out.params, x)

#minner = Minimizer(RectMinAttempt, params, fcn_args=(x, data))
#result = minner.minimize()
#final = data + result.residual
#report_fit(result)

#try:
#    import matplotlib.pyplot as plt
#    plt.plot(x, data, 'k+')
#    plt.plot(x, final, 'r')
#    plt.show()
#except ImportError:
#    pass
# %%
# in order to work, need to ensure that PSIIContr has had a resid column
# made and the other datasets have been run after

number_of_bins = 90
# An example of three data sets to compare
labels = ["Tree", "Crop", "Grass", "Shrub", "Total"]
data_sets = [PSIICTree['resid'], PSIICCrop['resid'],
             PSIICGrass['resid'], PSIICShrub['resid'], PSIIContr['resid']]
# Computed quantities to aid plotting
hist_range = (-.75, .52)
binned_data_sets = [
    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
    for d in data_sets]
binned_maximums = np.max(binned_data_sets, axis=1)
x_locations = [0, 111, 222, 333, 444]
# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)
# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc 
    ax.barh(centers, binned_data, height=heights, left=lefts, align='center')

ax.set_xticks(x_locations)
ax.set_xticklabels(labels)
ax.set_ylabel("Residuals")
ax.set_xlabel("Data sets")
plt.show()

fvalue, pvalue = stats.f_oneway(PSIICTree['resid'], PSIICCrop['resid'],
                                PSIICGrass['resid'], PSIICShrub['resid'])
print(fvalue, pvalue)
# %%
number_of_bins = 90
# An example of three data sets to compare
labels = ["Tree", "Crop", "Grass", "Shrub", "Total"]
data_sets = [PSIICTree['resid'], PSIICCrop['resid'],
             PSIICGrass['resid'], PSIICShrub['resid'], PSIIContr['resid']]
# Computed quantities to aid plotting
hist_range = (-.75, .52)
binned_data_sets = [
    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
    for d in data_sets]
binned_maximums = np.max(binned_data_sets, axis=1)
x_locations = [0, 111, 222, 333, 444]
# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)
# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc 
    ax.barh(centers, binned_data, height=heights, left=lefts, align='center')

ax.set_xticks(x_locations)
ax.set_xticklabels(labels)
ax.set_ylabel("Residuals")
ax.set_xlabel("Data sets")
plt.show()

fvalue, pvalue = stats.f_oneway(PSIICTree['resid'], PSIICCrop['resid'],
                                PSIICGrass['resid'], PSIICShrub['resid'])
print(fvalue, pvalue)
# %%
PSIIcold = PSIIContr[PSIIContr['HeatMid'] < 0]
PSIIcool = PSIIContr[(PSIIContr['HeatMid'] > 0) & (PSIIContr['HeatMid'] < 10)]
PSIIsteady = PSIIContr[(PSIIContr['HeatMid'] > 10) & (PSIIContr['HeatMid'] < 35)]
PSIIwarm = PSIIContr[(PSIIContr['HeatMid'] > 35) & (PSIIContr['HeatMid'] < 45)]
PSIIhot = PSIIContr[PSIIContr['HeatMid'] > 45]

number_of_bins = 90
# An example of three data sets to compare
labels = ["Cold", "Cool", "Steady", "Warm", "Hot"]
data_sets = [PSIIcold['resid'], PSIIcool['resid'], PSIIsteady['resid'],
             PSIIwarm['resid'], PSIIhot['resid']]
# Computed quantities to aid plotting
hist_range = (-.75, .52)
binned_data_sets = [
    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
    for d in data_sets]
binned_maximums = np.max(binned_data_sets, axis=1)
x_locations = [0, 40, 80, 250, 290]
# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)
# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc 
    ax.barh(centers, binned_data, height=heights, left=lefts, align='center')

ax.set_xticks(x_locations)
ax.set_xticklabels(labels)
ax.set_ylabel("Residuals")
ax.set_xlabel("Data sets")
plt.show()

fvalue, pvalue = stats.f_oneway(PSIIcold['resid'], PSIIcool['resid'], PSIIsteady['resid'],
                                PSIIwarm['resid'], PSIIhot['resid'])
print(fvalue, pvalue)
# %%
# Time focused, same needs as above cell
PSIIzero = PSIIContr[PSIIContr['Time (h)'] == 0]
PSIIyoung = PSIIContr[(0 < PSIIContr['Time (h)']) & (PSIIContr['Time (h)'] < 48.1)]
PSIItweek = PSIIContr[(48 < PSIIContr['Time (h)']) & (PSIIContr['Time (h)'] < 336.1)]
PSIIold = PSIIContr[(PSIIContr['Time (h)'] > 336)]

number_of_bins = 90
# An example of three data sets to compare
labels = ["Zero", "2 Day", "2 week", "Old", "Total"]
data_sets = [PSIIzero['resid'], PSIIyoung['resid'], PSIItweek['resid'],
             PSIIold['resid'], PSIIContr['resid']]
# Computed quantities to aid plotting
hist_range = (-.75, .52)
binned_data_sets = [
    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
    for d in data_sets]
binned_maximums = np.max(binned_data_sets, axis=1)
x_locations = [0, 75, 150, 225, 300]
# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)
# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc 
    ax.barh(centers, binned_data, height=heights, left=lefts, align='center')

ax.set_xticks(x_locations)
ax.set_xticklabels(labels)
ax.set_ylabel("Residuals")
ax.set_xlabel("Data sets")
plt.show()

fvalue, pvalue = stats.f_oneway(PSIIzero['resid'], PSIIyoung['resid'], 
                                 PSIItweek['resid'], PSIIold['resid'])
print(fvalue, pvalue)
# %%
print(unique(PSIIContr['Clade A']))
PSIIEud = PSIIContr[PSIIContr['Clade A'] == 'Eudicots']
PSIIMon = PSIIContr[PSIIContr['Clade A'] == 'Monocots']

number_of_bins = 90
# An example of three data sets to compare
labels = ["Eudicots", "Monocots", "Total"]
data_sets = [PSIIEud['resid'], PSIIMon['resid'], PSIIContr['resid']]
# Computed quantities to aid plotting
hist_range = (-.75, .52)
binned_data_sets = [
    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
    for d in data_sets]
binned_maximums = np.max(binned_data_sets, axis=1)
x_locations = [0, 130, 260]
# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)
# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc 
    ax.barh(centers, binned_data, height=heights, left=lefts, align='center')
ax.set_xticks(x_locations)
ax.set_xticklabels(labels)
ax.set_ylabel("Residuals")
ax.set_xlabel("Data sets")
plt.show()

fvalue, pvalue = stats.f_oneway(PSIIEud['resid'], PSIIMon['resid'])
print(fvalue, pvalue)
# %%
print(unique(PSIIContr['Clade B']))
PSIIRose = PSIIContr[PSIIContr['Clade B'] == 'Rosids']
PSIIComm = PSIIContr[PSIIContr['Clade B'] == 'Commelinids']
PSIIAste = PSIIContr[PSIIContr['Clade B'] == 'Asterids']
PSIIMagn = PSIIContr[PSIIContr['Clade B'] == 'Magnoliids']

number_of_bins = 90
# An example of three data sets to compare
labels = ["Rosids", "Commelinids", "Asterids", "Magnoliids", "Total"]
data_sets = [PSIIRose['resid'], PSIIComm['resid'],PSIIAste['resid'],
             PSIIMagn['resid'], PSIIContr['resid']]
# Computed quantities to aid plotting
hist_range = (-.75, .52)
binned_data_sets = [
    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
    for d in data_sets]
binned_maximums = np.max(binned_data_sets, axis=1)
x_locations = [0, 80, 160, 240, 320]
# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)
# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc 
    ax.barh(centers, binned_data, height=heights, left=lefts, align='center')
ax.set_xticks(x_locations)
ax.set_xticklabels(labels)
ax.set_ylabel("Residuals")
ax.set_xlabel("Data sets")
plt.show()

fvalue, pvalue = stats.f_oneway(PSIIRose['resid'], PSIIComm['resid'],
                                PSIIAste['resid'], PSIIMagn['resid'])
print(fvalue, pvalue)
# %%
PFabales = PSIIContr[PSIIContr['Order'] == 'Fabales']
PPoales = PSIIContr[PSIIContr['Order'] == 'Poales']
PAsterales = PSIIContr[PSIIContr['Order'] == 'Asterales']
PSolanales = PSIIContr[PSIIContr['Order'] == 'Solanales']
PGentianales = PSIIContr[PSIIContr['Order'] == 'Gentianales']
PEricales = PSIIContr[PSIIContr['Order'] == 'Ericales']
PAsparagales = PSIIContr[PSIIContr['Order'] == 'Asparagales']
PZingiberales = PSIIContr[PSIIContr['Order'] == 'Zingiberales']
PFagales = PSIIContr[PSIIContr['Order'] == 'Fagales']
PArecales = PSIIContr[PSIIContr['Order'] == 'Arecales']
PDipsacales = PSIIContr[PSIIContr['Order'] == 'Dipsacales']
PLamiales = PSIIContr[PSIIContr['Order'] == 'Lamiales']
PBoraginales = PSIIContr[PSIIContr['Order'] == 'Boraginales']
PBrassicales = PSIIContr[PSIIContr['Order'] == 'Brassicales']
PRosales = PSIIContr[PSIIContr['Order'] == 'Rosales']
PMagnoliales = PSIIContr[PSIIContr['Order'] == 'Magnoliales']
PLaurales = PSIIContr[PSIIContr['Order'] == 'Laurales']
PCaryophyllales = PSIIContr[PSIIContr['Order'] == 'Caryophyllales']
PAquifoliales = PSIIContr[PSIIContr['Order'] == 'Aquifoliales']
PPinales = PSIIContr[PSIIContr['Order'] == 'Pinales']
PMalpighiales = PSIIContr[PSIIContr['Order'] == 'Malpighiales']
PTrochodendrales = PSIIContr[PSIIContr['Order'] == 'Trochodendrales']
PBuxales = PSIIContr[PSIIContr['Order'] == 'Buxales']
PPolypodiales = PSIIContr[PSIIContr['Order'] == 'Polypodiales']
PSapindales = PSIIContr[PSIIContr['Order'] == 'Sapindales']
PMyrtales = PSIIContr[PSIIContr['Order'] == 'Myrtales']
PMalvales = PSIIContr[PSIIContr['Order'] == 'Malvales']
PVitales = PSIIContr[PSIIContr['Order'] == 'Vitales']

number_of_bins = 90
# An example of three data sets to compare
labels = ['Fabales', 'Poales', 'Asterales', 'Solanales', 'Gentianales',
          'Ericales', 'Asparagales', 'Zingiberales', 'Fagales', 'Arecales',
          'Dipsacales', 'Lamiales', 'Boraginales', 'Brassicales', 'Rosales',
          'Magnoliales', 'Laurales', 'Caryophyllales', 'Aquifoliales',
          'Pinales', 'Malpighiales', 'Trochodendrales', 'Buxales',
          'Polypodiales', 'Sapindales', 'Myrtales', 'Malvales', 'Vitales']
data_sets = [PFabales['resid'], PPoales['resid'], PAsterales['resid'], 
                    PSolanales['resid'], PGentianales['resid'], PEricales['resid'],
                    PAsparagales['resid'], PZingiberales['resid'], PFagales['resid'],
                    PArecales['resid'], PDipsacales['resid'], PLamiales['resid'],
                    PBoraginales['resid'], PBrassicales['resid'], PRosales['resid'],
                    PMagnoliales['resid'], PLaurales['resid'], PCaryophyllales['resid'],
                    PAquifoliales['resid'], PPinales['resid'], PMalpighiales['resid'],
                    PTrochodendrales['resid'], PBuxales['resid'], PPolypodiales['resid'],
                    PSapindales['resid'], PMyrtales['resid'], PMalvales['resid'],
                    PVitales['resid']]
# Computed quantities to aid plotting
hist_range = (-.75, .52)
binned_data_sets = [
        np.histogram(d, range=hist_range, bins=number_of_bins)[0]
        for d in data_sets]
binned_maximums = np.max(binned_data_sets, axis=1)
x_locations = [0,20,40,60,80,100,120,140,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560]
# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)
# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc 
    ax.barh(centers, binned_data, height=heights, left=lefts, align='center')
ax.set_xticks(x_locations)
ax.set_xticklabels(labels)
ax.set_ylabel("Residuals")
ax.set_xlabel("Data sets")
plt.show()

fvalue, pvalue = stats.f_oneway(PFabales['resid'], PPoales['resid'], PAsterales['resid'], 
                    PSolanales['resid'], PGentianales['resid'], PEricales['resid'],
                    PAsparagales['resid'], PZingiberales['resid'], PFagales['resid'],
                    PArecales['resid'], PDipsacales['resid'], PLamiales['resid'],
                    PBoraginales['resid'], PBrassicales['resid'], PRosales['resid'],
                    PMagnoliales['resid'], PLaurales['resid'], PCaryophyllales['resid'],
                    PAquifoliales['resid'], PPinales['resid'], PMalpighiales['resid'],
                    PTrochodendrales['resid'], PBuxales['resid'], PPolypodiales['resid'],
                    PSapindales['resid'], PMyrtales['resid'], PMalvales['resid'],
                    PVitales['resid'])
print(fvalue, pvalue)

# %%
Zero_sans = PSIIzero.drop([1079, 1059, 1044, 2046, 356, 1847, 1842, 384, 2029, 346, 348, 347], axis=0)
Young_sans = PSIIyoung.drop([1174, 168, 1185, 1194, 1203, 1176, 1007, 979, 1035, 1008, 980, 1036, 1009, 981, 1191, 1200, 1175, 1195, 1177, 1773, 1772, 2289], axis=0)
Tweek_sans = PSIItweek.drop([1043, 1048, 1053, 2143, 43, 1744, 1709, 1729, 1713, 1733, 2145, 2049, 2047, 2048, 39, 53, 20, 2035, 1677], axis=0)
Old_sans = PSIIold.drop([1607, 4, 6, 2050, 2051, 2052, 2028, 2038], axis=0)
# %%
ClimateComparison = PSIIold
plt.plot(ClimateComparison['HeatMid'][ClimateComparison['Climate'] == 0], 
         ClimateComparison['phiPSIImax'][ClimateComparison['Climate'] == 0], 'o',
         label='Control', alpha=0.5)
plt.plot(ClimateComparison['HeatMid'][ClimateComparison['Climate'] == 1], 
         ClimateComparison['phiPSIImax'][ClimateComparison['Climate'] == 1], 'o',
         label='Wild', alpha=0.5)
plt.title('Analysis of Impact of Non-controlled Origins (T=3)')
plt.legend()
# %%

Pzero = PSIIContr[PSIIContr['Time (h)'] == 0]
Pyoung = PSIIContr[(0 < PSIIContr['Time (h)']) & (PSIIContr['Time (h)'] < 48.1)]
Ptweek = PSIIContr[(48 < PSIIContr['Time (h)']) & (PSIIContr['Time (h)'] < 336.1)]
Pold = PSIIContr[(PSIIContr['Time (h)'] > 336)]

Pzero1 = Pzero[Pzero['Climate'] == 0]
Pyoung1 = Pyoung[Pyoung['Climate'] == 0]
Ptweek1 = Ptweek[Ptweek['Climate'] == 0]
Pold1 = Pold[Pold['Climate'] == 0]
# %%
PFT_numbers(Pzero)
PFT_numbers(Pyoung)
PFT_numbers(Ptweek)
PFT_numbers(Pold)
PFT_numbers(Pzero1)
PFT_numbers(Pyoung1)
PFT_numbers(Ptweek1)
PFT_numbers(Pold1)
PFT_numbers(PSIIContr[PSIIContr['Climate'] == 1])
# %%
modelTry = Model(Trial_Reaction_Equ)
#params = modelTry.guess(PSIIContr['phiPSIImax'], x=PSIIContr['HeatMid'])
#resultsTry = modelTry.fit(PSIIContr['phiPSIImax'], params, x=PSIIContr['HeatMid'])
resultsTry = modelTry.fit(PSIIContr['phiPSIImax'], x=PSIIContr['HeatMid'], tnaught=20, n=0.6, EsubT=20)

resultsTry.plot_fit()
plt.show()
print(resultsTry.fit_report())
# %%
# Attempting to produce a heat-map style plot of all data
trial = plt.hist2d(PSIIContr['HeatMid'], PSIIContr['phiPSIImax'],
                   bins=[60,30], density=False, norm=colors.LogNorm())

fig, ax = plt.subplots(figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})
ax.imshow(trial[0], interpolation='gaussian', cmap='jet')
plt.show()
# %%
