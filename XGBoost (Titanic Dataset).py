import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from xgboost import XGBClassifier

df = pd.read_csv("train.csv")

df = df.drop(['Name','Ticket','Cabin'], axis=1)
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2})

df.fillna(df.median(), inplace=True)

x = df.drop("Survived", axis=1)
y = df["Survived"]

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

xgb = XGBClassifier(n_estimators=100,learning_rate=0.1, max_depth=3)
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)
y_prob = xgb.predict_proba(x_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("Feature Importance:", xgb.feature_importances_)