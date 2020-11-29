from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

data_train = pd.read_csv(open('perceptron-train.csv'), header=None)
data_test = pd.read_csv(open('perceptron-test.csv'), header=None)

X_train = data_train.iloc[:, 1:]
y_train = data_train.iloc[:, 0]
X_test = data_test.iloc[:, 1:]
y_test = data_test.iloc[:, 0]
clf = Perceptron(random_state=241)
clf.fit(X=X_train, y=y_train)
predict = clf.predict(X=X_test)
acc = accuracy_score(y_true=y_test, y_pred=predict)
print("Accuracy without normalize:", round(acc, 3))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf_scaled = Perceptron()
clf_scaled.fit(X=X_train_scaled, y=y_train)
predict_scaled = clf_scaled.predict(X=X_test_scaled)
acc_n = accuracy_score(y_true=y_test, y_pred=predict_scaled)
print("Accuracy with normalize:", round(acc_n, 3))
print("Diff:", round(acc_n-acc, 3))