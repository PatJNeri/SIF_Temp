# Concept Building for more advanced formation of models
# Using density distributions, 2-way 1-to-1 concepts, and
# (potentially unifying spline curve construction)

# Description of concept
# 1) Current modelling method uses OLS
#    This method is focused on 


# %%
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.stats as stt
import numpy as np
import pandas as pd
# %%
# some basic functions
def minmax(val_list):
    min_val = min(val_list)
    max_val = max(val_list)

    return (min_val, max_val)


# %%
# Generation of example data for the purpose of explanation
#print(os.getcwd())
PSIImaster = pd.read_excel('c:/Users/PJN89/temp_git/PSIImax-Master2-24.xlsx',
                           engine='openpyxl')
PSIImaster['HeatMid'] = (PSIImaster['HeatUp'] + PSIImaster['HeatDown'])/2
PSIImaster.name = 'PSII Master'

PSIIContr = PSIImaster[(PSIImaster['water status'] == 0) & 
                       (PSIImaster['nut status'] == 0)]
PSIIContr.name = 'PSII master Control'
# %%
# Method to visualize the shape and 1-to-1 distributions. 
# The independent variable (x-axis) allows us to observe if there are
# any regions of excess deficiency in the data gathered.
# In general this distribution should be one where the bulk of
# the range gathered in the study has a high enough minimum value
# of data points. (this base value I believe can be arbitrary based
# on the availibility of data).
# on the extremes

# Original example was taken from here:
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
# Modifications were added based on the following 2 sources
# https://matplotlib.org/stable/tutorials/intermediate/gridspec.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html#scipy.stats.rv_histogram
# %%
x = PSIIContr['HeatMid']
y = PSIIContr['phiPSIImax']

def scatter_hist(x, y, ax, ax_histx, ax_histy, ax_cdfx, ax_cdfy, crnrplt):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)
    ax.grid(True)

    # now determine nice limits by hand:
    binwidthx = (np.max(x) - np.min(x))/20
    binwidthy = (np.max(y) - np.min(y))/20

    binsx = np.arange(np.min(x), np.max(x) + binwidthx, binwidthx)
    binsy = np.arange(np.min(y), np.max(y) + binwidthy, binwidthy)
    # Generate 
    a, b = minmax(x)
    x1 = np.linspace(a, b, 400)
    c, d = minmax(y)
    y1 = np.linspace(c, d, 400)
    # Plot hist of both axis
    ax_histx.hist(x, bins=binsx, facecolor='g')
    ax_histy.hist(y, bins=binsy, orientation='horizontal', facecolor='r')
    # Generate summary data of hist
    his_disx = stt.rv_histogram(np.histogram(x, bins=binsx))
    his_disy = stt.rv_histogram(np.histogram(y, bins=binsy))
    # Generate 2 axis equal distributed percentage gridding (CAN BE ADJUSTED)
    xgrad = ygrad = np.linspace(0, 1, 10)
    x_distr = his_disx.ppf(xgrad)
    y_distr = his_disy.ppf(ygrad)    
    # Plot CDF of both axis hist distributions
    ax_cdfx.set_title('X axis Divisons')
    ax_cdfx.plot(x1, his_disx.cdf(x1), 'g')
    for i in xgrad:
        ax_cdfx.hlines(i, a, his_disx.ppf(i), colors='grey')
        ax_cdfx.vlines(his_disx.ppf(i), 0, i, colors='grey', linestyles='dashed')
    ax_cdfy.set_ylabel('Y axis Divisions', fontsize=12)
    ax_cdfy.yaxis.set_label_position("right")
    ax_cdfy.plot(his_disy.cdf(y1), y1, 'r')
    for i in ygrad:
        ax_cdfy.vlines(i, c, his_disy.ppf(i), colors='grey')
        ax_cdfy.hlines(his_disy.ppf(i), 0, i, colors='grey', linestyles='dashed')

    # Plot data and the lines of equal % coverage
    crnrplt.vlines(x_distr, c, d, colors='g')
    crnrplt.hlines(y_distr, a, b, colors='r')
    crnrplt.plot(x,y, '+', alpha=0.3, markersize=4)

fig = plt.figure(figsize=(8, 8))

gs = fig.add_gridspec(3, 3,  width_ratios=(7, 2, 3), height_ratios=(3, 2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.2, hspace=0.2)

ax = fig.add_subplot(gs[2, 0])
ax_histx = fig.add_subplot(gs[1, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[2, 1], sharey=ax)
ax_cdfx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_cdfy = fig.add_subplot(gs[2, 2], sharey=ax)
crnrplt = fig.add_subplot(gs[:-1, 1:], sharex=ax, sharey=ax)

fig.suptitle('Data frequency equal % divisions', fontsize='large')
scatter_hist(x, y, ax, ax_histx, ax_histy, ax_cdfx, ax_cdfy, crnrplt)

plt.show()
# %%
# Isolate newly divided regions for further analysis
# For later singling out, redetermining values

x = PSIIContr['HeatMid']
y = PSIIContr['phiPSIImax']
binwidthx = (np.max(x) - np.min(x))/20
binwidthy = (np.max(y) - np.min(y))/20
binsx = np.arange(np.min(x), np.max(x) + binwidthx, binwidthx)
binsy = np.arange(np.min(y), np.max(y) + binwidthy, binwidthy)
his_disx = stt.rv_histogram(np.histogram(x, bins=binsx))
his_disy = stt.rv_histogram(np.histogram(y, bins=binsy))
xgrad = ygrad = np.linspace(0, 1, 10)
x_distr = his_disx.ppf(xgrad)
y_distr = his_disy.ppf(ygrad)  
# %%
r = np.linspace(-32, 63, 400)
for i in range(0, len(y_distr)-1):
    a = y_distr[i]
    b = y_distr[i+1]
    n_data = PSIIContr[(PSIIContr['phiPSIImax'] > a) &
                       (PSIIContr['phiPSIImax'] < b)]
    his_disy = stt.rv_histogram(np.histogram(n_data['HeatMid'], 
                                bins=binsx))
    plt.hist(n_data['HeatMid'], bins=binsx, density=True)
    plt.plot(r, his_disy.pdf(r))
    plt.xlim([-32, 63])
    plt.show()
# %%
s = np.linspace(0, 1, 400)
for i in range(0, len(x_distr)-1):
    a = x_distr[i]
    b = x_distr[i+1]
    n_data = PSIIContr[(PSIIContr['HeatMid'] > a) &
                       (PSIIContr['HeatMid'] < b)]
    his_disx = stt.rv_histogram(np.histogram(n_data['phiPSIImax'], 
                                bins=binsx))
    plt.hist(n_data['phiPSIImax'], bins=binsy, density=True)
    plt.xlim([0,1])
    plt.show()


# %%
# Concept proof for isolating Bimodal dist. Problem I am foreseeing
# is that my current understanding of lmfit is too weak, and I cannot 
# yet think of an initial param system that will produce assured results.
from lmfit.models import GaussianModel

n_data = PSIIContr[(PSIIContr['phiPSIImax'] > .318) &
                    (PSIIContr['phiPSIImax'] < .541)]

data = PSIIContr[PSIIContr['phiPSIImax'] < 0.34].reset_index().sort_values(by='HeatMid')
# Need to change the data inputs to the bins and values of hist.
point_x = binsx[:-1] + binwidthx/2
bin_vals = np.histogram(n_data['HeatMid'], bins=binsx)[0]
plt.plot(point_x, bin_vals)

x = point_x
y = bin_vals
gauss1 = GaussianModel(prefix='g1_')
pars = gauss1.guess(y, x=x)

pars['g1_center'].set(value=-10, min=-20, max=10)
pars['g1_sigma'].set(value=10)
pars['g1_amplitude'].set(value=30, min=0)

gauss2 = GaussianModel(prefix='g2_')
pars.update(gauss2.make_params())

pars['g2_center'].set(value=50, min=40, max=60)
pars['g2_sigma'].set(value=10)
pars['g2_amplitude'].set(value=65, min=0)

mod = gauss1 + gauss2

init = mod.eval(pars, x=x)
out = mod.fit(y, pars, x=x)

print(out.fit_report(min_correl=0.5))

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
axes[0].plot(x, y, 'bo')
axes[0].plot(x, init, 'ko', label='initial fit')
axes[0].plot(x, out.best_fit, 'o', label='best fit')
axes[0].legend(loc='best')

comps = out.eval_components(x=x)
axes[1].plot(x, y, 'b')
axes[1].plot(x, comps['g1_'], 'g--', label='Gaussian component 1')
axes[1].plot(x, comps['g2_'], 'm--', label='Gaussian component 2')
axes[1].legend(loc='best')

fig.suptitle(i)
plt.show()

# %%
# Attempt to put it all together
# Currently doesnt work (at least the modelling part)
# Does the first few where it actually looks like 
# 2 gaussians. But later things get less straightforward.
# Need to perhaps avoid a full classic smooth function
# PDF, and deal with the values more specifically.

# %%
# proof of concept
x = PSIIContr['HeatMid']

def minmax(val_list):
    min_val = min(val_list)
    max_val = max(val_list)

    return (min_val, max_val)

a, b = minmax(x)
x1 = np.linspace(a, b, 400)

binwidthx = (np.max(x) - np.min(x))/20
binsx = np.arange(np.min(x), np.max(x) + binwidthx, binwidthx)
his = np.histogram(x, bins=binsx)
his_dis = stt.rv_histogram(his)

plt.hist(x, density=True, bins=binsx)
plt.plot(x1, his_dis.pdf(x1))
plt.plot(x1, his_dis.cdf(x1))

# %%
