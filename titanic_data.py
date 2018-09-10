# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 12:01:52 2018

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

data2 = pd.read_csv('titanic.csv')
print('First few observations')
data2.head()

data2 = data2.drop('index', axis=1)
number = LabelEncoder()
data2['class'] = number.fit_transform(data2['class'].astype(str))
data2['age'] = number.fit_transform(data2['age'].astype(str))
data2['sex'] = number.fit_transform(data2['sex'].astype(str))
data2['survived'] = number.fit_transform(data2['survived'].astype(str))
print('After preprocessing...')
data2.head()

train2, test2 = train_test_split(data2, test_size=0.2)
pred2 = ['class', 'age', 'sex']
x_train = train2[pred2].values
y_train = train2['survived'].values
x_test = test2[pred2].values
y_test = test2['survived'].values

print('Correlation Matrix')
corr = data2.corr()
plt.figure(figsize=(12, 7))
sns.heatmap(corr, annot=True)

print('\nLOGISTIC REGRESSION')
model1 = sklearn.linear_model.LogisticRegression()
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
print(np.mean(cross_val_score(modelchk1, x_train, y_train, cv=7)))

print("\nSUPPORT VECTOR MACHINES")
clf1 = LinearSVC(random_state=0)
clf1.fit(x_train, y_train)
prediction3 = clf1.predict(x_test)
print(confusion_matrix(y_test, prediction3))
clftest = LinearSVC(random_state=0)
print(np.mean(cross_val_score(clftest, x_train, y_train, cv=5)))

print("\nMULTILAYERED PERCEPTRON")
clf2 = MLPClassifier(solver='lbfgs')
clf2.fit(x_train, y_train)
prediction4 = clf2.predict(x_test)
print(confusion_matrix(y_test, prediction4))
modelclf1 = MLPClassifier(solver='lbfgs')
print(np.mean(cross_val_score(modelclf1, x_train, y_train, cv=5)))
