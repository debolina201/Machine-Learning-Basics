# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:09:25 2018

@author: Debolina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.linear_model
import seaborn as sns

data1 = pd.read_csv('Pima.csv')
print('First few observations')
print(data1.head())
data1 = data1.drop('index', axis=1)
number = LabelEncoder()
data1['type'] = number.fit_transform(data1['type'].astype(str))
print('After preprocessing...')
print(data1.head())
train1, test1 = train_test_split(data1, test_size=0.3)
pred = ['npreg', 'glu', 'bp', 'skin', 'bmi', 'ped', 'age']

print('Correlation Matrix')
corr = data1.corr()
plt.figure(figsize=(12, 7))
sns.heatmap(corr, annot=True)

print('Logistic Regression')
model1 = sklearn.linear_model.LogisticRegression()
x_train = train1[pred].values
y_train = train1['type'].values
x_test = test1[pred].values
y_test = test1['type'].values

model1.fit(x_train, y_train)
predicted1 = model1.predict(x_test)
print(confusion_matrix(y_test, predicted1))

modeltest1 = sklearn.linear_model.LogisticRegression()
print('Cross val score')
print(np.mean(cross_val_score(modeltest1, x_train, y_train, cv=5)))

print("\nRANDOM FOREST CLASSIFIER")
model2 = RandomForestClassifier()
model2.fit(x_train, y_train)
prediction2 = model2.predict(x_test)
print(confusion_matrix(y_test, prediction2))
modelchk1 = RandomForestClassifier()
print(np.mean(cross_val_score(modelchk1, x_train, y_train, cv=6)))

print("\nSUPPORT VECTOR MACHINES")
clf1 = LinearSVC(random_state=0)
clf1.fit(x_train, y_train)
prediction3 = clf1.predict(x_test)
print(confusion_matrix(y_test, prediction3))
clftest = LinearSVC(random_state=0)
print(np.mean(cross_val_score(clftest, x_train, y_train, cv=6)))

print("\nMULTILAYERED PERCEPTRON")
clf2 = MLPClassifier(solver='lbfgs')
clf2.fit(x_train, y_train)
prediction4 = clf2.predict(x_test)
print(confusion_matrix(y_test, prediction4))
modelclf1 = MLPClassifier(solver='lbfgs')
print(np.mean(cross_val_score(modelclf1, x_train, y_train, cv=7)))