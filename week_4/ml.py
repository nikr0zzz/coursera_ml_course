import pandas as pd
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

train_data = pd.read_csv('salary-train.csv')
train_data['FullDescription'] = train_data['FullDescription'].apply(lambda v: v.lower())
train_data['FullDescription'] = train_data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
train_data['LocationNormalized'].fillna('nan', inplace=True)
train_data['ContractTime'].fillna('nan', inplace=True)

test_data = pd.read_csv('salary-test-mini.csv')
test_data['FullDescription'] = test_data['FullDescription'].apply(lambda v: v.lower())
test_data['FullDescription'] = test_data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
test_data['LocationNormalized'].fillna('nan', inplace=True)
test_data['ContractTime'].fillna('nan', inplace=True)

vectorizer = TfidfVectorizer(min_df=5)
X_train = vectorizer.fit_transform(train_data['FullDescription'])
X_test = vectorizer.transform(test_data['FullDescription'])

dictvectorizer = DictVectorizer()
X_train_one_hot = dictvectorizer.fit_transform(train_data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_one_hot = dictvectorizer.transform(test_data[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([X_train, X_train_one_hot])
X_test = hstack([X_test, X_test_one_hot])
y_train = train_data['SalaryNormalized']

model = Ridge(alpha=1, random_state=241)
model.fit(X=X_train, y=y_train)
y_test = model.predict(X=X_test)
print(y_test)


