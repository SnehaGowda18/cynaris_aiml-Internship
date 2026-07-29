import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("students.csv")

print("First 5 Rows")
print(df.head())

# Features and Target
X = df[["Hours", "Attendance", "Assignments"]]
y = df["Result"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Decision Tree
# -------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree Accuracy:")
print(accuracy_score(y_test, dt_pred))

print("\nDecision Tree Report")
print(classification_report(y_test, dt_pred))

# Plot Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["Fail", "Pass"],
    filled=True
)

plt.savefig("outputs/decision_tree.png")
plt.show()

# -------------------------
# Random Forest
# -------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

print("\nRandom Forest Report")
print(classification_report(y_test, rf_pred))

print("\nCompleted Successfully!")