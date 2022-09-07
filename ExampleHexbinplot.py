"""
ExampleHexbinplot
Author: Vibha Mane
Date Created: September 19, 2018
Last Update:  September 13, 2021

Hexbin plot for very large data sets; also has histogram plots.
Data generated from Normal Distribution
"""
#******************************************************************************

import numpy as np
import matplotlib.pyplot as plt
#******************************************************************************

#***********************************
# Create samples from the normal distribution
mux = 0
sigmax = 1.0

samples = 50000

x = np.random.normal(mux, sigmax, samples)
y = (x * 3 + np.random.normal(size=samples)) * 5

#***********************************
# statistics of data
muxEst = np.mean(x)
varxEst = np.var(x)
muyEst = np.mean(y)
varyEst = np.var(y)

#***********************************
# Sactter plot
plt.scatter(x, y, c='purple')
plt.title('Scatter Plot of Simulated Data')
plt.axis([-3, 3, -50, 50])
plt.show()
 
#***********************************
# Hexbin plot
plt.hexbin(x, y, gridsize=(20,20), cmap='Purples')
plt.title('Hexbin Plot of Simulated Data')
plt.axis([-3, 3, -50, 50])
plt.show()


#***********************************
# Histogram and normal dist PDF (probability density function) plots
count, bins, patches = plt.hist(x, 30, density=True, facecolor=(0, .785, .339))

plt.plot(bins, 1/(sigmax * np.sqrt(2 * np.pi)) *
np.exp( - (bins - mux)**2 / (2 * sigmax**2) ), linewidth=2, color='r')

plt.title('Histogram of Simulated Data')
plt.grid(True)
plt.show()
