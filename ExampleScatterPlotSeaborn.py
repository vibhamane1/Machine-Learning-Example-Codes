"""
ExampleScatterPlotSeaborn
Author: Vibha Mane
Date Created: September 21, 2018
Last Update:  September 9, 2021

Scatter Plots with the "Seaborn" module.
"""
#******************************************************************************
# Import necessary libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#******************************************************************************

#******************************************************************************
# Reading data into the pandas DataFrame, from a .csv file; it assumes that the 
# first row is labels.
iris = pd.read_csv('C:/VibhaTeaching/MLCourses/Generic/DataSets/UCIRepository/Iris/IrisData.csv')

#****************************************
# In the following plots, we use plt.figure() and plt.show(),
# before and after each plot, to avoid the plots from overlapping.

#****************************************
# Scatter Plots with "pairplot"
plt.figure(1)
sns.pairplot(data=iris, hue="species", palette="husl")
plt.show()

#****************************************
# Scatter plots with two features at a time
plt.figure(2)
sns.relplot(data=iris, x="sepal_length", y="sepal_width", hue="species");
plt.show()

plt.figure(3)
sns.relplot( data=iris, x="petal_length", y="petal_width", hue="species");
plt.show()


#****************************************
# Contour plot with "jointplot"
plt.figure(4)
sns.jointplot(data=iris, x="sepal_width", y="petal_length", kind="kde", hue="species")
plt.show()
