"""
ExampleMVN1
Author: Vibha Mane
Date Created: February 28, 2021
Last Update:  October 9, 2021

This program generates simulated data from Multivariate Normal Distribution (MVN).
Two different MVNs are used to represent 2 classes. The data is then stored
in a pandas dataframe for "pairplot".
"""

#***********************************
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#******************************************************************************
# Generate data from MVN; two classes, three features
# For each class, select the the mean vector muC, correlation matrix corC
# and the diagonal elements of DG.

#******* Class I Parameters***********
muC1 = [0, 2, 3];

# Correlation matrix, must be symmetric; the diagonal elements take values 1.0
# and the off-diagonal elements take values in the range [-1.0, 1.0].
corC1 = [(1.0, 0.1, 0.2),
         (0.1, 1.0, 0.0),
         (0.2, 0.0, 1.0)];

#***********************************
# Compute covariance matrix

# This is where you would select sigmas (standard deviation) for each feature.
DG = np.diag([1, 1.6, 2.2])

temp = np.matmul(DG, corC1)
covC1 = np.matmul(temp, DG)

#******* Class II Parameters ***********
muC2 = [0, -2, -3];

# Correlation matrix, must be symmetric; the diagonal elements take values 1.0
# and the off-diagonal elements take values in the range [-1.0, 1.0].
corC2 = [(1.0, -0.1, 0.1),
         (-0.1, 1.0, 1.0),
         (0.1, 1.0, 1.0)];

#***********************************
# Compute covariance matrix

# This is where you would select sigmas (standard deviation) for each feature.
DG = np.diag([3.0, 3.0, 3.0])

temp = np.matmul(DG, corC2)
covC2 = np.matmul(temp, DG)

#***********************************
# Specify sample size for each class
NSamples1 = 100
NSamples2 = 100
         
#***********************************
# Create columns, with class labels
Class1 = np.ones((NSamples1,1))
Class2 = np.full((NSamples2,1), 2)

#***********************************
# Generate samples, and put it in a dataframe
data1 = np.random.multivariate_normal(muC1, covC1, NSamples1)
Cdata1 = np.c_[data1, Class1]

df1 = pd.DataFrame(Cdata1, columns=["X1", "X2", "X3", "labels"])

data2 = np.random.multivariate_normal(muC2, covC2, NSamples2)
Cdata2 = np.c_[data2, Class2]

df2 = pd.DataFrame(Cdata2, columns=["X1", "X2", "X3", "labels"])


#***********************************
# Concatenate the two sets of samples
dfM = df1.append(df2, ignore_index=True)


#***********************************
# Pairplot
plt.figure(0)
f0 = sns.pairplot(dfM, hue="labels", palette="Set2", diag_kind="hist")
f0.fig.suptitle("MVN Simulated Data", y=1.08, color="#0000ff", fontsize=15)
plt.show()
