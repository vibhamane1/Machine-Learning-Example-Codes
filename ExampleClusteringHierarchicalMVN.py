"""
ExampleClusteringHierarchicalMVN
Authors: Vibha Mane and Ravikiran Cherukuri
Date Created: October 22, 2020
Last Update:  September 11, 2021

This program generates simulated data from multivariate normal (MVN)
distribution; the data is then stored in a pandas dataframe for "pairplot".
We next implement hierarchical clustering with this data.
"""
#******************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.cluster.hierarchy as sch
import statistics

from IPython.core.pylabtools import figsize


#******************************************************************************
# Generate data from MVN; three classes, three features;
# define parameters for each class; these are the mean and the covariance
# matrices of MVN distribution

#******* Class I ***********
muC1 = [0, 2, 3]
covC1 = [[1.2, 0.4, 0.7], [0.4, 3.0, 0.0], [0.7, 0.0, 5.0]]

#******* Class II ***********
muC2 = [0, -2, -3]
covC2 = [[1.0, -0.4, 0.2], [-0.4, 1.0, 0.0], [0.2, 0.0, 1.0]]
     
#******* Class III ***********
muC3 = [0, 0, 0]
covC3 = [[0.5, 0.1, 0.2], [0.1, 0.5, 0.0], [0.2, 0.0, 0.5]]


#******* Sample Size ***********
NSamples1 = 10
NSamples2 = 10
NSamples3 = 10
         
#****************************************
# Create columns, with class labels;
# class labes are used for visualization
Class1 = np.ones((NSamples1,1))
Class2 = np.full((NSamples2,1), 2)
Class3 = np.full((NSamples3,1), 3)

#****************************************
# Generate samples from MVN; add a column of class labels; put this into
# a dataframe
data1 = np.random.multivariate_normal(muC1, covC1, NSamples1)
Cdata1 = np.c_[data1, Class1]
df1 = pd.DataFrame(Cdata1, columns=["X1", "X2", "X3", "labels"])

data2 = np.random.multivariate_normal(muC2, covC2, NSamples2)
Cdata2 = np.c_[data2, Class2]
df2 = pd.DataFrame(Cdata2, columns=["X1", "X2", "X3", "labels"])

data3 = np.random.multivariate_normal(muC3, covC3, NSamples3)
Cdata3 = np.c_[data3, Class3]
df3 = pd.DataFrame(Cdata3, columns=["X1", "X2", "X3", "labels"])


# Concatenate the three sets of samples
dfT = df1.append(df2, ignore_index=True)
dfM = dfT.append(df3, ignore_index=True)


#***********************************
# Scatter Plots with "pairplot"
plt.figure(0)
sns.pairplot(dfM, hue="labels", palette="Set2")
plt.title("MVN Simulated Data", color="g")
plt.show()

#****************************************
# Separate X (features) from Y (target label)
dfMX = dfM[["X1", "X2", "X3"]]


#****************************************
# Perform heirarchical clustering, with the X features; note that the target variable
# is not utilized in clustering.

# linkage
Z=sch.linkage(dfMX, method="average", metric="euclidean")

# cutoff
list=[Z[-2,2], Z[-3,2]]
cutoff = statistics.median(list)


#***********************************
# Graphical representation 
figsize(10,10)
plt.figure(1)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("sample index")
plt.ylabel("distance")
sch.dendrogram(Z, color_threshold=4.5)
plt.axhline(y=cutoff, color='c', linestyle='--')
plt.show()
