# fitting ang predicting : estimator basics
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[1,2,3],
     [11,12,13]]
y = [0,1]

clf.fit(X,y)
RandomForestClassifier(random_state=0)
clf.predict(X)

# transformers and pre-processors
from sklearn.preprocessing import StandardScaler
X = [[0,15],
     [1,-10]]
StandardScaler().fit(X).transform(X)

