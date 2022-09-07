"""
ExamplePCA1
Author: Vibha Mane
Date Created: February 27, 2021
Last Update:  October 20, 2021

Generate 2-class data from Multivariate Normal Distribution (MVN) and
plot pairplots. Next, do feature reduction with PCA
(Principal Component Analysis). Look at the value of "Explained Variance" of
the principal components and select number of components such that the sum of
explained variance is large enough. Scree plots are also shown.
"""

#***********************************
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
# create columns, with class label
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
# Plot histograms
plt.figure(0)
dfM.hist(column=["X1","X2", "X3"], bins=20, facecolor=(0, .785, .339))
plt.show()

#***********************************
# Pairplot
plt.figure(1)
f1 = sns.pairplot(dfM, hue="labels", diag_kind="hist", palette="Set2")
f1.fig.suptitle("MVN Simulated Data", y=1.08, color="#0000ff", fontsize=15)
plt.show()


#***********************************
# Standardizing the features
dfMPart = dfM[["X1", "X2", "X3"]]

ArraySS = StandardScaler().fit_transform(dfMPart) 


#****************************************
# Perform PCA
pca = PCA(n_components=3)
PCAData = pca.fit_transform(ArraySS)

#****************************************
# Explained Variance
print("\n")
print("PCA Explained Variance")
print(pca.explained_variance_ratio_)
print("\n")

#****************************************
# Scree plot
PC_values = np.arange(pca.n_components_)

plt.figure(1)
plt.bar(PC_values, pca.explained_variance_ratio_, width=0.4, color='salmon')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()


#****************************************
# DataFrame after extracting Principal Components
columns = ["PC1", "PC2", "PC3"]
dfMPCA = pd.DataFrame(data = PCAData, columns = columns)

dfMPCA["labels"] = dfM["labels"]


#***********************************
# Pairplot
plt.figure(2)
f2 = sns.pairplot(dfMPCA, hue="labels", diag_kind="hist", palette="Set2")
f2.fig.suptitle("After PCA", y=1.08, color="#0000ff", fontsize=15)
plt.show()
