# run code that gets out the range of the data
if //range.low// < //gold.low//:
    # data extends beyond low side of gold zone
    # run a low side model
    set C2 = 50
    set S2 = 2
    # adjust dataset to be low only
    newSet = oldSet[oldSet['temp'] < gold.high]
    # run model to find C1, S1, and Amp
if //range.high// > //gold.high//:
    # data extends above high side of gold zone
    # run a high side model
    set C1 = -6
    set S1 = 2
    # adjust dataset to be high only
    newSet = oldSet[oldSet['temp'] > gold.low]
    # run model to find C2, S2, and Amp

# This in theory should produce 2 models for each side, with the limit
# being the far side of the gold zone relative to hot or cold run.

# r-squared values are important to take from this, as well as the
# grey-line 1-sigma plots. Important to figure out how the values
# are used to make the 1-sigma line, so that it can be applied to
# the below plots for isolated or tandem plots.

# For a plotting comparison, perhaps I could invert and overlay the 
# 2 models on top of each other, relative to the C points the model
# gives. The idea being this would be a showing of the difference in
# slope (S) and Amp, while visualizing how the actual data aligns if 
# you separate using this method. Could also be used as a method to 
# refine a better defined gold zone.

ColdModelNumbers = [cC1, cS1, cC2, cS2, cAmp]
HotModelNumbers = [hC1, hS1, hC2, hS2, hAmp]
# Recall here that we have already defined 4 of these values by hand.
# Then perhaps we center around C both datasets
hotSet['tempnew'] = hotSet['temp'] - hC2
coldSet['tempnew'] = coldSet['temp'] - cC1
# The parts of the whole rectangle model that are relevant are now 
# facing opposite directions. One of them needs to be inverted 
# around the new Zero-point. I think visually inverting the hot
# model will look best.

hotrange = np.linspace(hotMin, hotMax, 500)
coldrange = np.linspace(coldMin, coldMax, 500)
# make 2 datasets of the model function values
hotModel = 
coldModel = 

# So first plot the 2 datasets.
plt.plot(hotSet['phiPSIImax'], hotSet['tempnew'], 'o')
plt.plot(coldSet['phiPSIImax'], coldSet['tempnew'], 'o')
# Then plot the 2 modelled equations
plt.plot(hotModel, hotrange, '-r', label='High Temp Model')
plt.plot(coldModel, coldrange, '-c', label='Cold Temp Model')

# Also consider potentially making a 2 subplot graphic with the
# two model plots, that include grey output values?
# %%
# 8-26
# Formal attempt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import RectangleModel
import scipy.stats as stt
import math
# %%
# get the data
print(os.getcwd())
PSIImaster = pd.read_excel('PSIImax-Master2-24.xlsx')
PSIImaster['HeatMid'] = (PSIImaster['HeatUp'] + PSIImaster['HeatDown'])/2
PSIImaster['Heatrange'] = PSIImaster['HeatUp'] - PSIImaster['HeatDown']
PSIImaster.name = 'PSII Master'

PSIIContr = PSIImaster[(PSIImaster['water status'] == 0) & (PSIImaster['nut status'] == 0)]
PSIIContr.name = 'PSII master Control'

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
# function to grab data range

def minmax(val_list):
    min_val = min(val_list)
    max_val = max(val_list)

    return (min_val, max_val)

def Phunction_Name_Here(data):
    """Make sure such that data has a HeatMid column
    value and that it has a Fv/Fm column called phiPSIImax"""

    # Define the Gold Zone
    goldrange = (10, 30)
    # Find the range classification
    datarange = minmax(data['HeatMid'])  # Get the range of temps
    if datarange[0] < goldrange[0]:
        newdata = data[data['HeatMid'] < goldrange[0]] # Determine if low temp is possible
        print(len(newdata['HeatMid']))
        if len(newdata['HeatMid']) < 20:  # arbitrary, unless stats i dont know??
            print('WARNING: Low data range is small. Results may vary.')
            choice1 = input('Would you like to proceed? [y/n]')
            if choice1 == 'y':
                newdata1 = data[data['HeatMid'] < goldrange[1]] # include any goldrange data for better amp estimate
                Ordered = newdata1.sort_values(by='HeatMid')
                x = Ordered['HeatMid']
                x0 = x.iloc[:]
                y = Ordered['phiPSIImax']
                y0 = y.iloc[:]
                # Quad Model run
                mod = RectangleModel(form='erf')
                pars = mod.guess(y0, x=x0)
                pars['amplitude'].set(value=0.8, min=0.6, max=0.83)
                pars['center1'].set(value=-6, min=-12, max=7)
                pars['center2'].set(value=46, vary=False)
                pars['sigma1'].set(value=7, min=1, max=12)
                pars['sigma2'].set(value=2, vary=False)
                outCool = mod.fit(y, pars, x=x)
                print(outCool.fit_report())
            else:
                print('Low range not modelled')
        else:
            newdata1 = data[data['HeatMid'] < goldrange[1]] # include any goldrange data for better amp estimate
            Ordered = newdata1.sort_values(by='HeatMid')
            x = Ordered['HeatMid']
            x0 = x.iloc[:]
            y = Ordered['phiPSIImax']
            y0 = y.iloc[:]
            # Quad Model run
            mod = RectangleModel(form='erf')
            pars = mod.guess(y0, x=x0)
            pars['amplitude'].set(value=0.8, min=0.6, max=0.83)
            pars['center1'].set(value=-6, min=-12, max=7)
            pars['center2'].set(value=46, vary=False)
            pars['sigma1'].set(value=7, min=1, max=12)
            pars['sigma2'].set(value=2, vary=False)
            outCool = mod.fit(y, pars, x=x)
            print(outCool.fit_report())
    if datarange[1] > goldrange[1]:
        newdata2 = data[data['HeatMid'] > goldrange[1]] # Determine if high temp is possible
        print(len(newdata2['HeatMid']))
        if len(newdata2['HeatMid']) < 20:  # arbitrary, unless stats i dont know??
            print('WARNING: High data range is small. Results may vary.')
            choice2 = input('Would you like to proceed? [y/n]')
            if choice2 == 'y':
                newdata3 = data[data['HeatMid'] > goldrange[0]] # include any goldrange data for better amp estimate
                Ordered = newdata3.sort_values(by='HeatMid')
                x = Ordered['HeatMid']
                x0 = x.iloc[:]
                y = Ordered['phiPSIImax']
                y0 = y.iloc[:]
                # Quad Model run
                mod = RectangleModel(form='erf')
                pars = mod.guess(y0, x=x0)
                pars['amplitude'].set(value=0.8, min=0.6, max=0.83)
                pars['center1'].set(value=-6, vary=False)
                pars['center2'].set(value=46, min=35, max=57)
                pars['sigma1'].set(value=2, vary=False)
                pars['sigma2'].set(value=7, min=1, max=12)
                outHot = mod.fit(y, pars, x=x)
                print(outHot.fit_report())
            else:
                print('High range not modelled')
        else:
            newdata3 = data[data['HeatMid'] > goldrange[0]] # include any goldrange data for better amp estimate
            Ordered = newdata3.sort_values(by='HeatMid')
            x = Ordered['HeatMid']
            x0 = x.iloc[:]
            y = Ordered['phiPSIImax']
            y0 = y.iloc[:]
            # Quad Model run
            mod = RectangleModel(form='erf')
            pars = mod.guess(y0, x=x0)
            pars['amplitude'].set(value=0.8, min=0.6, max=0.83)
            pars['center1'].set(value=-6, vary=False)
            pars['center2'].set(value=46, min=35, max=57)
            pars['sigma1'].set(value=2, vary=False)
            pars['sigma2'].set(value=7, min=1, max=12)
            outHot = mod.fit(y, pars, x=x)
            print(outHot.fit_report())        
    if (datarange[0] > goldrange[0]) & (datarange[1] < goldrange[1]):
        print('Data range does not stray outside of gold range. Cannot be modelled successfully.')


# %%
i = input('chose a PFT')
print('This run is for Adjusted PFT ' + str(i))
Phunction_Name_Here(PSIIContr[PSIIContr['Adjusted PFT'] == i])
# %%
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
# 9-2-21
# The following is an attempt to narrow in on a better system to
# determine the amplitude of the various models

# Step 1
# check the total numerical mean of all data in the 'central range'

def gold_range_mean(dataset, range=[10,30]):
    constrainrangedata = dataset[(dataset['HeatMid'] > range[0]) &
                                 (dataset['HeatMid'] < range[1])]
    plt.hist(constrainrangedata['phiPSIImax'])
    plt.show()
    leng = len(constrainrangedata['HeatMid'])
    avg = np.mean(constrainrangedata['phiPSIImax'])
    med = np.median(constrainrangedata['phiPSIImax'])
    std = np.std(constrainrangedata['phiPSIImax'])
    kurt = stt.kurtosis(constrainrangedata['phiPSIImax'], fisher=True)

    return [leng, avg, med, std, kurt]


# %%
# Adjust what is in the [] to select a specific output for all PFT's
for i in range(1,17):
    print('This run is for Adjusted PFT ' + str(i))
    # Can add an index if one specific element is needed
    print(gold_range_mean(PSIIContr[PSIIContr['Adjusted PFT'] == i])[4])


# %%




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

def get_data_plot_modAmp(dataset):
        """ Pick a chosen dataset that has a HeatMid column and phiPSIImax column
        This function uses the mid range data metrics to pick an Amp"""
        h = int(input('Choose which output of gold_range_mean to use'))
        Ordered = dataset.sort_values(by='HeatMid')
        x = Ordered['HeatMid']
        x0 = x.iloc[:]
        y = Ordered['phiPSIImax']
        y0 = y.iloc[:]
        # Quad Model run
        mod = RectangleModel(form='erf')
        pars = mod.guess(y0, x=x0)
        if h == 0:
            pars['amplitude'].set(value=0.8, min=0.6, max=0.83)
        else:
            pars['amplitude'].set(value=gold_range_mean(dataset)[h], vary=False)
        pars['center1'].set(value=0, min=-12, max=7)
        pars['center2'].set(value=46, min=35, max=57)
        pars['sigma1'].set(value=7, min=1, max=12)
        pars['sigma2'].set(value=5, min=1, max=12)
        out = mod.fit(y, pars, x=x)
        ModelChoice = 'Rect' #input('Chose between [Quad, Rect] or [Both] models:')
        if ModelChoice == 'Rect':
                #print('You have chosen Rectangle model for the ', dataset.name, 'dataset.')
                print(out.fit_report())
                ps = get_Mod_paramsValues(out)
                #print(ps.val[:])
                if h == 0:
                    A, m1, s1, m2, s2 = ps.val[0], ps.val[1], ps.val[2], ps.val[3], ps.val[4]
                else:
                    A, m1, s1, m2, s2 = gold_range_mean(dataset)[1], ps.val[0], ps.val[1], ps.val[2], ps.val[3]
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
                correlation_matrix = np.corrcoef(y, Ayy)
                correlation_xy = correlation_matrix[0,1]
                r_squared = correlation_xy**2
                print(r_squared)
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
                plt.annotate('$\mathregular{R^{2}}$ - ' + str(round(r_squared, 2)) + '\nN = ' + str(y.shape[0]), xy=(1,1),
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
                plt.show()
                #plt.savefig(dataset.name + 'Box5-28-21.JPG')  


# %%
