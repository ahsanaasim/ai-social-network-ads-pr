
from knn import KNN
from svm import SVM
from lr import LR
from nb import NB
from dt import DT
from rf import RF

fileName = "fay_data_2.json"
fileName2 = "Social_Network_Ads.csv"

k = KNN(fileName2, type="csv")
k.build()
k.predict()
k.accuracy()

s = SVM(fileName2, type="csv")
s.build()
s.predict()
s.accuracy()

ks = SVM(fileName2, type="csv")
ks.build('rbf')
ks.predict()
ks.accuracy()


l = LR(fileName2, type="csv")
l.build()
l.predict()
l.accuracy()


nb = NB(fileName2, type="csv")
nb.build()
nb.predict()
nb.accuracy()


dt = DT(fileName2, type="csv")
dt.build()
dt.predict()
dt.accuracy()

rf = RF(fileName2, type="csv")
rf.build()
rf.predict()
rf.accuracy()



