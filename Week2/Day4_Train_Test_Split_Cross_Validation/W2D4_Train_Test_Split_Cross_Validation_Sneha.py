import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Sample Dataset
# -----------------------------
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [60,65,70,75,80,85,90,92,95,98],
    "Pass": [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["Hours", "Attendance"]]
y = df["Pass"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

# -----------------------------
# Cross Validation
# -----------------------------
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())

# Save Results
with open("outputs/cross_validation_scores.txt", "w") as f:
    f.write("Cross Validation Scores\n")
    f.write(str(scores))
    f.write("\n")
    f.write(f"\nAverage Accuracy: {scores.mean()}")

print("\nResults saved successfully!")