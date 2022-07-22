# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from lmfit.models import RectangleModel

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

def get_setdata_plotttt(dataset):
        """ Pick a chosen dataset that has a HeatMid column and phiPSIImax column"""
        Ordered = dataset.sort_values(by='HeatMid')
        x = Ordered['HeatMid']
        x0 = x.iloc[:]
        y = Ordered['phiPSIImax']
        y0 = y.iloc[:]
        # Quad Model run
        mod = RectangleModel(form='erf')
        pars = mod.guess(y0, x=x0)
        pars['amplitude'].set(value=0.8, min=0.6, max=0.83)
        pars['center1'].set(value=-6, min=-23, max=7)
        pars['center2'].set(value=46, min=35, max=57)
        pars['sigma1'].set(value=7, min=1, max=25)
        pars['sigma2'].set(value=5, min=1, max=12)
        out = mod.fit(y, pars, x=x)
        ModelChoice = 'Rect' #input('Chose between [Quad, Rect] or [Both] models:')
        if ModelChoice == 'Rect':
                Tittt = input('Give title of plot')
                #print('You have chosen Rectangle model for the ', dataset.name, 'dataset.')
                print(out.fit_report())
                ps = get_Mod_paramsValues(out)
                #print(ps.val[:])
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
                # Various indexes
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
                print('R-squared : ', r2_score(y, Ayy))
                print('MAE : ', MAE)
                print('RMSE : ', RMSE)
                print('Wilm : ', Wilm)                
                
                # Below is the attempt at formatting a boxplot figure (can be optimized)
                TrialBox = dataset.sort_values(by='HeatMid')
                
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
                plt.figure(3)
                plt.rcParams["figure.figsize"] = (10,10)
                plt.boxplot([Box0['phiPSIImax'], Box1['phiPSIImax'], Box2['phiPSIImax'], Box3['phiPSIImax'],
                             Box4['phiPSIImax'], Box5['phiPSIImax'], Box6['phiPSIImax'], Box7['phiPSIImax'],
                             Box0_5['phiPSIImax'], Box1_5['phiPSIImax'], Box2_5['phiPSIImax'], Box3_5['phiPSIImax'],
                             Box4_5['phiPSIImax'], Box5_5['phiPSIImax'], Box6_5['phiPSIImax'], Box7_5['phiPSIImax']],
                             positions=(-14.5, -4.5, 5.5, 15.5, 25.5, 35.5, 45.5, 55.5,
                                        -9.5, 0.5, 10.5, 20.5, 30.5, 40.5, 50.5, 60.5), widths=5)
                plt.plot(x, y, 'o', alpha=0.3)
                x2 = np.linspace(-32, 63, 500)
                Aa12 = (x2 - m1)/s1
                Aaa12 = []
                for i in range(len(Aa12)):
                        Aaa12.append(math.erf(Aa12[i]))
                Aa22 = -(x2 - m2)/s2
                Aaa22 = []
                for i in range(len(Aa22)):
                        Aaa22.append(math.erf(Aa22[i]))
                Ayy2 = []
                for i in range(len(Aaa12)):
                       Ayy2.append((A/2)* (Aaa12[i] + Aaa22[i]))
                plt.plot(x2, Ayy2, 'r-')
                #plt.plot(x, y, 'k+', alpha=0.6)
                #plt.plot(x, out.best_fit, 'r-', label='best fit')
                #dely = out.eval_uncertainty(sigma=1)
                #plt.fill_between(x, out.best_fit-dely, out.best_fit+dely, color="#ABABAB",
                #                 label='1-$\sigma$ \nuncertainty band')
                
                plt.annotate('$\mathregular{R^{2}}$ = ' + str(round(r2_score(y, Ayy), 2)) + '\nN = ' + str(len(y.index)), xy=(1,1),
                             xycoords='axes fraction', xytext=(-10, -10), textcoords='offset pixels',
                             horizontalalignment='right',
                             verticalalignment='top', fontsize=17)
                plt.xticks(ticks=range(-32,68, 5), labels=range(-32,68, 5), fontsize=10)
                plt.yticks(np.linspace(0,1,6), fontsize = 10)
                plt.ylabel('Maximum Quantum Efficiency of PSII\n (Fv/Fm)', fontsize=17)
                plt.xlabel('Experiment Temperature ' + u'\u2103', fontsize=17)
                #plt.title(dataset.name + ' Boxplot (width 10) with data')
                plt.ylim([0, 1])
                plt.xlim([-32, 63])
                plt.grid(True)
                plt.title(Tittt, fontsize=20)

# %%
