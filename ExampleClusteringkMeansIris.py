"""
ExampleClusteringkMeansIris
Author: Vibha Mane
Date Created: September 28, 2018
Last Update:  September 11, 2021

Scatter Plots with the "seaborn" module.
"""
#******************************************************************************
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
#******************************************************************************


#***********************************
# Load iris data into a pandas DataFrame, from the built-in Seaborn Data Set
iris = sns.load_dataset("iris")

#****************************************
# In the following plots, we use plt.figure() and plt.show(),
# before and after each plot, to avoid the plots from overlapping.

#****************************************
# Scatter Plots with "pairplot"
plt.figure(1)
sns.pairplot(data=iris, hue="species", palette="husl")
plt.show()


#****************************************
# Separate X (features) and Y (target variable)
irisX = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
irisY = iris[["species"]]

#****************************************
# Perform k-means clustering, with the X features; note that the target variable
# is not utilized in clustering; n_clusters is an input parameter.
km = KMeans(n_clusters=3)
km.fit(irisX)

#****************************************
# Extract cluster centers
centers = km.cluster_centers_
print(centers)

#****************************************
# Extract cluster labels
iris["new_labels"] = km.labels_
# Map species names to numbers for scatter plots
iris["species"] = iris["species"].map({"setosa":0, "versicolor":1, "virginica":2})

#****************************************
# Plot original classes and the new clusters.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
ax[0].scatter(iris["sepal_length"], iris["sepal_width"], c=iris["species"], cmap="autumn", s=150)
ax[1].scatter(iris["sepal_length"], iris["sepal_width"], c=iris["new_labels"], cmap="winter", s=150)

ax[0].set_xlabel('Sepal length', fontsize=15)
ax[0].set_ylabel('Sepal width', fontsize=15)
ax[0].set_title('Before Clustering', fontsize=15, color="g")

ax[1].set_xlabel('Sepal length', fontsize=15)
ax[1].set_ylabel('Sepal width', fontsize=15)
ax[1].set_title('After Clustering', fontsize=15, color="g")

plt.show()

