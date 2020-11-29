import pandas as pd
import numpy as np
from sklearn.svm import SVC

data = pd.read_csv("svm-data.csv", header=None)
y_train = data.iloc[:, 0]
X_train = data.iloc[:, 1:]

clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(X=X_train, y=y_train)
sv = np.sort(clf.support_)
print("Indexes of support vectors")
print(' '.join(map(str, clf.support_+1)))
with open("1.txt", "w") as file:
    file.write(' '.join(map(str, clf.support_+1)))
