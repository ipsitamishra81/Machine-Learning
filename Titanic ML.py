import pandas as pd
data = pd.read_csv("C:/Users/HP/train.csv")
print(data.head())

print(data.info())
print(data.describe())
print(data.isnull().sum())

print(data['Survived'].value_counts())

data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
data = data.fillna(data.median(numeric_only=True))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)

data['Sex'] = data['Sex'].map({'male':0, 'female': 1})

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data['Embarked'] = data['Embarked'].fillna(0)

x = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']]
y = data['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

print("Accuracy:", model.score(x_test, y_test))