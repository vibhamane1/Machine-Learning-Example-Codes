"""
Spyder Editor
ExampleHistPlot
Author: Vibha Mane
Date Created: September 9, 2021
Last Update:  September 9, 2021

Histogram and Kernel Density Plots, with the "seaborn" Module
Some plots are with the Iris data set loaded from a .csv file;
some other plots are with simulated data generated from a Mixture Normal
Distribution. Also illustrated is how to save a plot to a .pdf file.
"""
#******************************************************************************
# Import necessary libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#******************************************************************************

#******************************************************************************
# Reading data into the pandas DataFrame, from a .csv file; it assumes that the 
# first row is labels.
df = pd.read_csv('C:/VibhaTeaching/MLCourses/Generic/DataSets/UCIRepository/Iris/IrisData.csv')

#****************************************
# In the following plots, we use plt.figure() and plt.show(),
# before and after each plot, to avoid the plots from overlapping.

#****************************************
# Histogram plot of a specific feature from the DataFrame df 
plt.figure(1)
sns.histplot(data=df, x="sepal_length", bins=20, color="c")
plt.title("Histogram, All species", color="c")
plt.show()

#****************************************
# Histogram plot of a specific feature, with hue set to target variable
plt.figure(2)
sns.histplot(data=df, x="sepal_length", bins=20, hue="species")
plt.title("Histogram, species in different colors", color="b")
plt.show()


#****************************************
# Next, we create three different data frames for the three values of
# target variable, and plot the histogram of each.
# Histogram plot of a specific feature, with hue set to target variable
setosa = df[df["species"]=="Iris-setosa"]
versicolor = df[df["species"]=="Iris-versicolor"]
virginica = df[df["species"]=="Iris-virginica"]

plt.figure(3)
sns.histplot(data=setosa, x="sepal_length", bins=10, color="m")
plt.title("Histogram, species=setosa", color="m")
plt.show()

plt.figure(4)
sns.histplot(data=versicolor, x="sepal_length", bins=10, color="m")
plt.title("Histogram, species=versicolor", color="m")
plt.show()

plt.figure(5)
sns.histplot(data=virginica, x="sepal_length", bins=8, color="m")
plt.title("Histogram, species=virginica", color="m")
plt.show()

#****************************************
# Next, generate simulated data from the mixture normal distribution
# (using NumPy)
mu1 = 0
sigma1 = 1.0
mu2 = 4.0
sigma2 = 1.5

cf = 0.3
SamplesTotal = 1000
SamplesN1 = int(cf*SamplesTotal)
SamplesN2 = int(SamplesTotal-SamplesN1)

x1 = np.random.normal(mu1, sigma1, SamplesN1)
x2 = np.random.normal(mu2, sigma2, SamplesN2)
x1 = np.append(x1 ,x2)        
    
#****************************************
# Histogram and Kernel Density Plots with "distplot"; note the handel fa in
# the command distplot(); this is used for the pdf save.
plt.figure(6)
fa=sns.displot(x1, bins=40, kde=True, color="g")
plt.title("Histogram and Kernel Density Estimate, Mixture Normal", color="g")
plt.grid(True)
plt.show()

#****************************************
# Save the plot to a PDF file
fa.savefig('Example1.pdf')
