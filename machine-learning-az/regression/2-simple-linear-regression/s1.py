# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:54:40 2018

@author: Tapiwa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m


dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

regressor.score(X_test, y_test)
y_pred = regressor.predict(X_test)

# Visualize

plt.scatter(X_train, y_train, c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')
plt.title('Salary vs Experience (Training Set). Score ' + str(round(regressor.score(X_train, y_train), 3)))
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_train, y_train, c='red')
plt.plot(X_test, regressor.predict(X_test), c='blue')
plt.title('Salary vs Experience (Test Set). Score ' + str(round(regressor.score(X_train, y_train), 3)))
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()
