# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

class NB:
    
    def __init__(self, filepath):
        # Importing dataset
        self.dataset = pd.read_json(filepath)
        self.X = self.dataset.iloc[:, :-1].values
        self.y = self.dataset.iloc[:, -2].values
        self.preProcess()
        
    
    def handleMissingData(self):
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(self.X[:, 1:2])
        self.X[:, 1:2] = imputer.transform(self.X[:, 1:2])

        imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputer2.fit(self.X[:, 0:1])
        self.X[:, 0:1] = imputer2.transform(self.X[:, 0:1])
        
    def encodeCategoricals(self):
        ct = make_column_transformer((OneHotEncoder(), [0]), (OneHotEncoder(), [2]), (OneHotEncoder(), [3]), remainder='passthrough', sparse_threshold=0)
        t = ct.fit_transform(self.X)
        self.X = np.array(t)
        
    def encodeDependants(self):
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)
        
    def splitTestNTrains(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 1)

    def scaleFeatures(self):
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.fit_transform(self.X_test)
        
    def preProcess(self):
        self.handleMissingData()
        self.encodeCategoricals()
        self.encodeDependants()
        self.splitTestNTrains()
        self.scaleFeatures()
        
    def build(self):
        self.classifier = GaussianNB()
        self.classifier.fit(self.X_train, self.y_train)
        
    def predict(self):
        self.y_pred = self.classifier.predict(self.X_test)
        
    def accuracy(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print("Naive Bayes Accuracy: ", accuracy*100)


