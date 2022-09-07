"""
ExampleDbscan1
Author: Vibha Mane
Date Created: September 28, 2018
Last Update:  September 11, 2021

DBSCAN clustering algorithm with data from the UCI Repository
"""
#******************************************************************************
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

#******************************************************************************
# Reading data into the pandas DataFrame, from a .csv file; it assumes that the 
# first row is labels.
df = pd.read_csv('C:/VibhaTeaching/MLCourses/Generic/DataSets/UCIRepository/WholesaleCustomers/WholesaleCustomersData.csv')

#****************************************
# The following command displays all columns
pd.set_option('display.expand_frame_repr', False)

# Print some stuff
print(df.columns.values)
print(df.ndim)
print(df.shape)

# Check for missing data
print(df.info())

# Descriptive statistics with "pandas"
dStats = df.describe()

#****************************************
# Keep a few features
dfM = df[["Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]]


#****************************************
# Perform DBSCAN clustering; choose the parameters eps and MinPts
epsA = 2000
MinPts = 7
db = DBSCAN(eps=epsA, min_samples=MinPts).fit(dfM)

#****************************************
# Extract cluster labels
labels = db.labels_
NClusters = len(set(labels))

#****************************************
# Create a column of the cluster labels in the DataFrame
dfM["Cluster"] = labels

#****************************************
# Scatter Plots with "Cluster" as hue
plt.figure(1)
fa = sns.pairplot(data=dfM, hue="Cluster", palette="Set2")
fa.set(xlim=(0, 20000), ylim=(0,20000))
plt.show()

#****************************************
# Save the plot to PDF file
fa.savefig('ExampleCluster.pdf')
