import pandas as pd
from sklearn import metrics

data = pd.read_csv('classification.csv')
y_true = data['true']
y_pred = data['pred']

data_score = pd.read_csv('scores.csv')

tp, tn, fp, fn = 0, 0, 0, 0

for i in range(len(data)):
    if data['true'][i] == 1 and data['pred'][i] == 1:
        tp += 1
    if data['true'][i] == 1 and data['pred'][i] == 0:
        fn += 1
    if data['true'][i] == 0 and data['pred'][i] == 0:
        tn += 1
    if data['true'][i] == 0 and data['pred'][i] == 1:
        fp += 1

print('TP = {0}, FP = {1}, FN = {2}, TN = {3}'.format(tp, fp, fn, tn))

acc = metrics.accuracy_score(y_true, y_pred)
prec = metrics.precision_score(y_true, y_pred)
rec = metrics.recall_score(y_true, y_pred)
f = metrics.f1_score(y_true,y_pred)

print('Accuracy = {0}, Precision = {1}, Recall = {2}, F = {3}'.format(round(acc, 2), round(prec,2), round(rec, 2), round(f, 2)))

scores = {}
for x in data_score.columns[1:]:
    scores[x] = metrics.roc_auc_score(data_score['true'], data_score[x])
print(scores)

scores = {}
for x in data.columns[1:]:
    curve = metrics.precision_recall_curve(data['true'], data[x])
    df = pd.DataFrame({'precision': curve[0], 'recall':curve[1]})
    scores[x] = df[df['recall'] >= 0.7]['precision'].max()
print(scores)
