import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

# Load dataset
df = pd.read_csv("diabetes.csv")

# Handle missing values
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols] = df[cols].replace(0, pd.NA)
df.fillna(df.median(), inplace=True)

# Features
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1 (Full tree)
dt1 = DecisionTreeClassifier(random_state=42)
dt1.fit(X_train, y_train)

y_pred1 = dt1.predict(X_test)

print("=== Full Tree ===")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred1))
print("Report:\n", classification_report(y_test, y_pred1))

# Model 2 (Limited depth)
dt2 = DecisionTreeClassifier(max_depth=3, random_state=42)
dt2.fit(X_train, y_train)

y_pred2 = dt2.predict(X_test)

print("\n=== Depth=3 Tree ===")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred2))
print("Report:\n", classification_report(y_test, y_pred2))

# Feature Importance
print("\nFeature Importance:", dt2.feature_importances_)