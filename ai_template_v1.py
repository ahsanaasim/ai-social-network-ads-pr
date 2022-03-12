### import necessary thingies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
###

### file import
filepath = 'Data.csv'
dataset = pd.read_csv(filepath)

# filepath = "ourTestData.json"
# data = pd.read_json(filepath)
###

### Defining our axes
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
###

### Separating test and train datasets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
###

### Scaling values to minimal
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
###

