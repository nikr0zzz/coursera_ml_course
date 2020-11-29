import pandas as pd
import math
from sklearn.metrics import roc_auc_score


def fw1(w1, w2, y, X, k, C):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[1][i] * (1.0 - 1.0 / (1.0 + math.exp(-y[i] * (w1*X[1][i] + w2*X[2][i]))))

    return w1 + (k * (1.0 / l) * S) - k * C * w1


def fw2(w1, w2, y, X, k, C):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[2][i] * (1.0 - 1.0 / (1.0 + math.exp(-y[i] * (w1*X[1][i] + w2*X[2][i]))))

    return w2 + (k * (1.0 / l) * S) - k * C * w2


def grad(y, X, C=0.0, w1=0.0, w2=0.0, k=0.1, err=1e-5):
    i = 0
    i_max = 10000
    w1_new, w2_new = w1, w2

    while True:
        i += 1
        w1_new, w2_new = fw1(w1, w2, y, X, k, C), fw2(w1, w2, y, X, k, C)
        e = math.sqrt((w1_new - w1) ** 2 + (w2_new - w2) ** 2)

        if i >= i_max or e <= err:
            break
        else:
            w1, w2 = w1_new, w2_new

    return [w1_new, w2_new]


def a(X, w1, w2):
    return 1.0 / (1.0 + math.exp(-w1*X[1]-w2*X[2]))


data = pd.read_csv('data-logistic.csv', header=None)
X = data[[1, 2]]
y = data[0]

w1, w2 = grad(y=y, X=X, C=0)
rw1, rw2 = grad(y=y, X=X, C=10.0)

y_score = X.apply(lambda x: a(x, w1, w2), axis=1)
y_rscore = X.apply(lambda x: a(x, rw1, rw2), axis=1)

auc = roc_auc_score(y_true=y, y_score=y_score)
rauc = roc_auc_score(y_true=y, y_score=y_rscore)

print(round(auc, 3), round(rauc, 3))