"""
ExampleKernelPCA1
Author: Vibha Mane
Date Created: February 27, 2021
Last Update:  February 27, 2021

Kernel PCA  (Principal Component Analysis), with two concentric circles data
"""

#***********************************
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles


#***********************************
# create circles data
X, Y = make_circles(n_samples=100, noise=0.1, random_state=123, factor=0.5)

plt.scatter(X[Y==0, 0], X[Y==0, 1], color='red', alpha=0.5)
plt.scatter(X[Y==1, 0], X[Y==1, 1], color='blue', alpha=0.5)

plt.title('Circles Data Set')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


#***********************************
# Perform Kernel PCA
kpca = KernelPCA(n_components=5, kernel='rbf', gamma=1)
KPCAData=kpca.fit_transform(X)

# from array to DataFrame
columns = ['KPCA1', 'KPCA2', 'KPCA3', 'KPCA4', 'KPCA5']
dfMKPCA = pd.DataFrame(data = KPCAData, columns = columns)

# Find proportion of variance, before adding a "labels" col to DataFrame
covKPCA = dfMKPCA.cov()
print(covKPCA)

dfMKPCA["labels"] = Y

#***********************************
# Scatter Plots
plt.figure(2)
sns.pairplot(dfMKPCA, hue="labels", diag_kind="hist", palette="husl")
