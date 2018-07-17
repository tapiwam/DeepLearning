# -*- coding: utf-8 -*-
"""
Regression Template

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
y = dataset.iloc[:, 2].values

# =========================
# >>>> NOT ENOUGH DATA TO SPLIT INTO TRAINING SET AND TESTING SET
"""
#Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# =========================
# Feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# =========================
# Model/Fitting regression model 

# regressor =

# =========================
# Predict a new result with regression

y_pred = regressor.predict(6.5)

# =========================
# Visualize the training set results

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color='blue' )
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Job Level')
plt.ylabel('Salary')
plt.show()

# =========================
# Visualize the training set results -> Higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
 
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X_grid), color='blue' )
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Job Level')
plt.ylabel('Salary')
plt.show()
