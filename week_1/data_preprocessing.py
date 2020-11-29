import pandas
import re
def clean_name(name):
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)
    name = name.split(' ')[0].replace('"', '')
    return name



data = pandas.read_csv('titanic/train.csv', index_col='PassengerId')
lenght = len(data)
males = len(data[data.Sex == 'male'])
females = lenght - males
f = open('1.txt', 'w')
f.write(str(males)+' '+ str(females))
f.close()

survived = len(data[data.Survived == 1])
survived = round(survived/lenght*100, 2)
f = open('2.txt', 'w')
f.write(str(survived))
f.close()

first_class_passengers = len(data[data.Pclass == 1])
first_class_passengers = round(first_class_passengers/lenght*100, 2)
f = open('3.txt', 'w')
f.write(str(first_class_passengers))
f.close()

avg = round(data.Age.mean(),2)
med = round(data.Age.median(),2 )
f = open('4.txt', 'w')
f.write(str(avg)+' '+str(med))
f.close()

corr = round(data.corr().loc['SibSp', 'Parch'], 2)
f = open('5.txt', 'w')
f.write(str(corr))
f.close()

fem_names = data[data['Sex'] == 'female']['Name'].apply(clean_name)
name_counts = fem_names.value_counts()
f = open('6.txt', 'w')
f.write(name_counts.head(1).index.values[0])
f.close()