# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
filepath = 'fay_data.json'
dataset = pd.read_json(filepath)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -2].values
# print("Importing dataset")
# print(X)

# X_df = pd.DataFrame(X)
# y_df = pd.DataFrame(y)

# Take care missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:2])
X[:, 1:2] = imputer.transform(X[:, 1:2])

imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer2.fit(X[:, 0:1])
X[:, 0:1] = imputer2.transform(X[:, 0:1])

print("Take care missing data")
print(X)

# Encode Categorical Values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

ct = make_column_transformer((OneHotEncoder(), [0]), remainder='passthrough', sparse_threshold=0)

#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0]), ], remainder='passthrough', sparse_threshold=0)
t = ct.fit_transform(X)
X = np.array(t)

# print("Encode Categorical Values")
# print(X)


# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Training Logistic Regression Model 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
print("Predicting the Test set results")
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
print("Making the Confusion Matrix")
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy*100)






