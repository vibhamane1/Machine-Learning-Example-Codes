"""
ExampleSVM
Author: Vibha Mane
Date Created: February 28, 2021
Last Update:  November 4, 2021

Generate 2-class data from Multivariate Normal Distribution (MVN),
with non-linear boundaries; perform data visualization with scatter plots.
Next, perform SVM (Support Vector Machine) classification and plot decision
regions. Also print some accuracy measures, such as confusion matrix and
classification report.
"""

#***********************************
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import svm

from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from IPython.core.pylabtools import figsize

#******************************************************************************
# Generate data from MVN; two classes, two features, non-linear separation

#******* Class I Parameters***********
# Class I data is generated from two different MVNs
muC11 = [-3, -5];
covC11 = [(1.0, 0.2),
         (0.2, 1.0)];

muC12 = [3, 5];
covC12 = [(1.0, 0.2),
         (0.2, 1.0)];

NSamples11 = 50
NSamples12 = 50

#******* Class II Parameters ***********
muC2 = [0, 0];
covC2 = [(2.0, 0.1),
         (0.1, 2.0)];

NSamples2 = 50
         
#***********************************
# Create columns, with class label
Class11 = np.ones((NSamples11,1))   # Class I
Class12 = np.ones((NSamples12,1))   # Class I

Class2 = np.full((NSamples2,1), 2)  # Class II
      
#***********************************
# Generate samples, and put it in a dataframe
data11 = np.random.multivariate_normal(muC11, covC11, NSamples11)
Cdata11 = np.c_[data11, Class11]

df11 = pd.DataFrame(Cdata11, columns=["X1", "X2", "labels"])

data12 = np.random.multivariate_normal(muC12, covC12, NSamples12)
Cdata12 = np.c_[data12, Class12]

df12 = pd.DataFrame(Cdata12, columns=["X1", "X2", "labels"])

data2 = np.random.multivariate_normal(muC2, covC2, NSamples2)
Cdata2 = np.c_[data2, Class2]

df2 = pd.DataFrame(Cdata2, columns=["X1", "X2", "labels"])

# Concatenate the three sets of samples
dfMM = df11.append(df12, ignore_index=True)
dfM = dfMM.append(df2, ignore_index=True)


#****************************************
# Some plots
figsize(6,6)

#***********************************
# Pairplot
plt.figure(0)
f0 = sns.pairplot(dfM, hue="labels", palette="Set2", diag_kind="hist")
f0.fig.suptitle("MVN Simulated Data", y=1.08, color="#0000ff", fontsize=15)
plt.show()

#***********************************
# Split data into train/test set
TrainData, TestData = train_test_split(dfM, test_size=0.3, shuffle=True)

#***********************************
# Pairplot Train Data
plt.figure(1)
f1 = sns.pairplot(TrainData, hue="labels", palette="Set2", diag_kind="hist")
f1.fig.suptitle("Train Data", y=1.08, color="#0000ff", fontsize=15)
plt.show()

#***********************************
# Pairplot Test Data
plt.figure(2)
f2 = sns.pairplot(TestData, hue="labels", palette="Set2", diag_kind="hist")
f2.fig.suptitle("Test Data", y=1.08, color="#0000ff", fontsize=15)
plt.show()

#***********************************
# Separate X (features) and Y (target variable)
TrainDataX = TrainData[["X1", "X2"]]
TrainDataY = TrainData["labels"]

TestDataX = TestData[["X1", "X2"]]
TestDataY = TestData["labels"]

#***********************************
# Perform SVM on Train Data -
# You can try different kernels here;
# since the classes are non-lineraly separated, we need a non-linear kernel
#clf = svm.SVC(kernel='linear', degree=3)
#clf = svm.SVC(kernel='poly', degree=10)
clf = svm.SVC(kernel='rbf', gamma=0.7)
clf.fit(TrainDataX, TrainDataY)


#***********************************
# Perform Prediction on Test Data; note that we use features (X) for
# prediction, and class labels (Y) for comparison.
PredictY = clf.predict(TestDataX)
PredictYM = PredictY.astype(int) # convert to int, used for plot Decision Regions

# Series to array
ArrayTestX = TestDataX.to_numpy()
ArrayTestY = TestDataY.to_numpy()


#***********************************
# Compare predicted class labels with true class labels
print("\n")
print("True Class Labels")
print(ArrayTestY)
print("Predicted Class Labels")
print(PredictY)

# Print confusion matrix
CF = confusion_matrix(ArrayTestY, PredictY)
print("\n")
print("Confusion Matrix Test Set")
print(CF)

# Print classification report
print("\n")
print("Classification Report Test Set")
CP = classification_report(ArrayTestY, PredictY)
print(CP)


#***********************************
# Plot Decision Regions for Test Data
plt.figure(3)
plot_decision_regions(ArrayTestX, PredictYM, clf=clf, legend=2)
plt.title("Decision Regions", color='#0000ff', fontsize=15)
plt.show()