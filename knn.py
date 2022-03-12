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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

class KNN:
    
    def __init__(self, filepath, type="json"):
        # Importing dataset
        if type == "json":
            self.dataset = pd.read_json(filepath)
        elif type == "csv":
            self.dataset = pd.read_csv(filepath)
            
        self.X = self.dataset.iloc[:, :-1].values
        self.y = self.dataset.iloc[:, -1].values
        self.handleMissingData()
        # self.encodeCategoricals()
        self.encodeDependants()
        self.splitTestNTrains()
        self.scaleFeatures()
        
    
    def handleMissingData(self):
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(self.X[:, 1:2])
        self.X[:, 1:2] = imputer.transform(self.X[:, 1:2])

        
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
        
    def build(self):
        self.classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        self.classifier.fit(self.X_train, self.y_train)
        print(self.classifier.predict(self.sc.transform([[30,87000]])))
        
    def predict(self):
        self.y_pred = self.classifier.predict(self.X_test)
        
    def accuracy(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print("KNN Accuracy: ", accuracy*100)
        
    def plot(self):
        print("plot")
        self.X_set, self.y_set = self.sc.inverse_transform(self.X_train), self.y_train
        self.X1, self.X2 = np.meshgrid(np.arange(start = self.X_set[:, 0].min() - 10, stop = self.X_set[:, 0].max() + 10, step = 1),
                             np.arange(start = self.X_set[:, 1].min() - 1000, stop = self.X_set[:, 1].max() + 1000, step = 1))
        plt.contourf(self.X1, self.X2, self.classifier.predict(self.sc.transform(np.array([self.X1.ravel(), self.X2.ravel()]).T)).reshape(self.X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(self.X1.min(), self.X1.max())
        plt.ylim(self.X2.min(), self.X2.max())
        for i, j in enumerate(np.unique(self.y_set)):
            plt.scatter(self.X_set[self.y_set == j, 0], self.X_set[self.y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('K-NN (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
        print("plotted")





