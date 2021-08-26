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