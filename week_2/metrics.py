from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn import datasets
import pandas as pd
import numpy as np

data = datasets.load_boston()
X = data.data
y = data.target
X_scale = scale(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
kmean = list()
for t in np.linspace(1,10,num=200):
    kn = KNeighborsRegressor(n_neighbors=5, weights='distance', p=t)
    array = cross_val_score(estimator=kn, X=X_scale, y=y, cv=kf, scoring='neg_mean_squared_error')
    kmean.append(array.mean())
    print(np.average(array))

print(max(kmean),'p =', kmean.index(max(kmean))+1)
