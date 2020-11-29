import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'] )
X_train = newsgroups.data
y_train = newsgroups.target

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X=X_train_vec, y=y_train)

C = gs.best_params_['C']

clf = SVC(kernel= 'linear', C=C, random_state= 241)
clf.fit(X_train_vec, y_train)

indexes = np.argsort(np.abs(clf.coef_.toarray()[0]))[-10:]
print(indexes)
words = np.array(vectorizer.get_feature_names())
print(words[indexes])

print(' '.join(np.sort(words[indexes])))
with open("1.txt", "w") as file:
  file.write(' '.join(np.sort(words[indexes])))




