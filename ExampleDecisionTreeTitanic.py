"""
ExampleDecisionTreeTitanic
Author: Vibha Mane
Date Created: Jan 19, 2020
Last Update:  November 4, 2021

This example illustrates decision tree classifier with the Titanic data set
Input features are mostly categorical. Further, the categorical features are
mapped into numbers.
"""
#******************************************************************************

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from IPython.core.pylabtools import figsize

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#******************************************************************************
# Reading data into the pandas DataFrame, from a csv file;
# it assumes that the first row is labels;

df = pd.read_csv('C:/VibhaTeaching/MLCourses/Generic/DataSets/UCIRepository/Titanic/TitanicData.csv')
df.head()


#***********************************
# Map categorical values to numeric, as the decision tree classifier
# uses numbers for categories; note that this is simply category to
# numerical mapping, not one hot encoding
df["Sex"] = df["Sex"].map({"female": 0, "male": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S":2})


#***********************************
# Keep a few features
dfM = df[["Pclass", "Sex", "SibSp", "Parch", "Fare", "Survived", "Embarked"]]

#***********************************
# Drop the rows with any missing values
dfMD = dfM.dropna()
print(dfMD.info())

#***********************************
# Split data into train/test set
TrainData, TestData = train_test_split(dfMD, test_size=0.3, shuffle=True)

#***********************************
# Separate X (features) and Y (target variable)

input_features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]

TrainDataX = TrainData[input_features]
TrainDataY = TrainData["Survived"]

TestDataX = TestData[input_features]
TestDataY = TestData["Survived"]


#***********************************
# Decision Tree Classification - train the classifier with the train data set
model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
model.fit(TrainDataX, TrainDataY)


#***********************************
# Predict the response for test data set
PredictY = model.predict(TestDataX)

# Series to array
ArrayTestY = TestDataY.to_numpy()

#***********************************
# Compare predicted class labels with true class labels
# print("True Class Labels")
# print(ArrayTestY)
# print("Predicted Class Labels")
# print(PredictY)

# Print confusion matrix
CF = confusion_matrix(ArrayTestY, PredictY)
print("Confusion Matrix")
print(CF)

# Print classification report
CP = classification_report(ArrayTestY, PredictY)
print(CP)


#***********************************
# Visualizing the decision tree
cn=["Not Survived", "Survived"]

figsize(15,15)
plt.figure(0)
tree.plot_tree(model)
tree.plot_tree(model, feature_names = input_features, class_names=cn, filled = True);
plt.title("Decision Tree with Titatic Data Set", color='#0000ff')
plt.show()