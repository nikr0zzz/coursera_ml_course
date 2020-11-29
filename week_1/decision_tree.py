import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic/train.csv', usecols=['Survived', 'Pclass', 'Age', 'Fare', 'Sex'])
X_test = pd.read_csv('titanic/test.csv', usecols=['Pclass', 'Age', 'Fare', 'Sex'])
sex_transform = {'male': 1, 'female': 0}
data['Sex'] = data['Sex'].map(sex_transform)
data = data.dropna()
X_test = X_test.dropna()
X_test['Sex'] = X_test['Sex'].map(sex_transform)
Y = data['Survived']
X = data.drop(['Survived'], axis='columns')

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, Y)
importances = pd.Series(clf.feature_importances_,index = list(X))
fout = open('out.txt', 'w')
fout.write(' '.join(importances.sort_values(ascending=False).head(2).index.values))
pred = clf.predict(X_test)
