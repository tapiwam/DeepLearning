# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 06:32:16 2018

@author: tapiw
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# =========================
#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# =========================
# Categorical Entries -> Country and Purchased
# Encode categorical text for models

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# =========================
# Avoiding the dummy variable trap
X = X[:, 1:]


# =========================
#Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =========================
# Feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# =========================
# Modelling