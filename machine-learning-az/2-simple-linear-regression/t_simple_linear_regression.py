# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 20:08:46 2018

@author: tapiw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
#Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# See what the data looks like
# dataset.plot()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# =========================
#Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# =========================
# Fit the data to a linear model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# =========================
# Predict the Test results

# Vector of predictions using the training model
y_pred = regressor.predict(X_test)

# =========================
# Visualize the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue' )
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# =========================
# Visualize the test set results
plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X_test, y_pred, color='blue' ) -> Model the same whether we use test or train set
plt.plot(X_train, regressor.predict(X_train), color='blue' )
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


