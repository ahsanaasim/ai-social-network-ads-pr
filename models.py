
from knn import KNN
from svm import SVM
from lr import LR
from nb import NB
from dt import DT
from rf import RF

fileName = "fay_data_2.json"
fileName2 = "Social_Network_Ads.csv"

k = KNN(fileName, type="json")
k.build()
k.predict()
k.accuracy()

# s = SVM(fileName)
# s.build()
# s.predict()
# s.accuracy()

# ks = SVM(fileName)
# ks.build('rbf')
# ks.predict()
# ks.accuracy()


# l = LR(fileName)
# l.build()
# l.predict()
# l.accuracy()


# nb = NB(fileName)
# nb.build()
# nb.predict()
# nb.accuracy()


# dt = DT(fileName)
# dt.build()
# dt.predict()
# dt.accuracy()

# rf = RF(fileName)
# rf.build()
# rf.predict()
# rf.accuracy()



