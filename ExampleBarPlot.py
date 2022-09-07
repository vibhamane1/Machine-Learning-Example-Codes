"""
ExampleBarPlot
Author: Vibha Mane
Date Created: September 25, 2018
Last Update:  November 4, 2021

Utilizing the "Seaborn" Module to plot categorical data;
Bar plots with the Titanic data set.
Also illustrated is how to save a DataFrame to a .csv file.
"""

#******************************************************************************
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.core.pylabtools import figsize

#***********************************
# Load titanic data into a pandas DataFrame, from the built-in Seaborn data set
titanic = sns.load_dataset("titanic")

#***********************************
# Display some stuff
# The following command displays all columns
pd.set_option('display.expand_frame_repr', False)
# slice some rows
#print(titanic[0:8])

# The following command displays the first few rows
print("First few rows of data")
print(titanic.head())
print("\n\n")

# The following command displays the last few rows
print("Last few rows of data")
print(titanic.tail())

# The following command writes dataframe to a .csv file 
titanic.to_csv('output.csv')

#***********************************
figsize(7,7)

#***********************************
# Create a bar plot for the target variable
plt.figure(0)
sns.countplot(data=titanic, x="survived")
plt.show()

#***********************************
# Create a bar plot using "catplot"
plt.figure(1)
sns.catplot(data=titanic, x="class", y="survived", hue="sex", palette="husl", kind="bar", legend=True)               
plt.show()

#***********************************
# Some more bar plots with "countplot"
plt.figure(2)
sns.countplot(data=titanic, x="survived", hue="class", palette="husl")
plt.show()

plt.figure(3)
sns.countplot(data=titanic, x="survived", hue="sex", palette="husl")
plt.show()

plt.figure(4)
sns.countplot(data=titanic, x="survived", hue="embarked", palette="husl")
plt.show()

plt.figure(5)
sns.countplot(data=titanic, x="survived", hue="sibsp", palette="husl")
plt.show()
