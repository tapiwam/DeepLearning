# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:18:07 2018

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
# Fitting Linear regression -> For comparisson

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_1 = lin_reg.predict(X)

# =========================
# Model polinomial regression
from sklearn.preprocessing import PolynomialFeatures

# Degrees
poly_degree = 4

## Transform into quadratic polinomial set
poly_reg = PolynomialFeatures(degree=poly_degree)
X_poly = poly_reg.fit_transform(X)

# Fit using Linear regression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
y_pred_2 = lin_reg_2.predict(X_poly)


# =========================
# Visualize the training set results

plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred_1, color='blue' )
plt.plot(X, y_pred_2, color='purple' )
plt.title('Job Level vs Salary')
plt.xlabel('Job Level')
plt.ylabel('Salary')
plt.show()

# =========================
# Visualize the training set results - > More granular

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='purple' )
plt.title('Job Level vs Salary')
plt.xlabel('Job Level')
plt.ylabel('Salary')
plt.show()

# Predict a new result with linear regression
lin_reg.predict(6.5)

# Predict a new result with polinomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

