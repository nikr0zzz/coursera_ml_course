from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np

data = pd.read_csv('wine.data')
Y = data['0']
X = data.iloc[:, 1:]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kMeans = list()
for k in range(1,51):
    kn = KNeighborsClassifier(n_neighbors=k)
    array = cross_val_score(estimator=kn, X=X, y=Y, cv=kf, scoring='accuracy')
    m = array.mean()
    kMeans.append(m)

m = max(kMeans)
print(round(m,2), kMeans.index(m)+1)

X_scale = scale(X)
kMeans = list()
for k in range(1,51):
    kn = KNeighborsClassifier(n_neighbors=k)
    array = cross_val_score(estimator=kn, X=X_scale, y=Y, cv=kf, scoring='accuracy')
    m = array.mean()
    kMeans.append(m)

m = max(kMeans)
print(round(m,2), kMeans.index(m)+1)
print(X_scale)