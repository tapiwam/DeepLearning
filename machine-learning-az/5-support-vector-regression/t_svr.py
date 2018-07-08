# -*- coding: utf-8 -*-
"""
Regression - svr

@author: tapiw
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# =========================
#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

#X = dataset.iloc[:, :-1].values
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# =========================
# >>>> NOT ENOUGH DATA TO SPLIT INTO TRAINING SET AND TESTING SET
"""
#Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# =========================
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# =========================
# Model/Fitting regression model 
from sklearn.svm import SVR
#regressor = SVR(kernel='rbf', degree=6, C=2**5)
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# =========================
# Predict a new result with regression

y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
sc_y.inverse_transform(y_pred)

# =========================
# Visualize the training set results

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color='blue' )
# plt.plot(X, y_pred, color='blue' )
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Job Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color='blue' )
# plt.plot(X, y_pred, color='blue' )
plt.title('Truth or Bluff (SVR Scaled Model)')
plt.xlabel('Job Level')
plt.ylabel('Salary')
plt.show()

