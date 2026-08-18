import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =====================================
# Create outputs folder
# =====================================
os.makedirs("outputs", exist_ok=True)

# =====================================
# Load Dataset
# =====================================
df = pd.read_csv("diabetes.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

# =====================================
# Features and Target
# =====================================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =====================================
# Train Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# Train Model
# =====================================
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# =====================================
# Print Coefficients
# =====================================
print("\nIntercept")
print(model.intercept_)

print("\nCoefficients")

coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print(coef)

# =====================================
# Prediction
# =====================================
y_pred = model.predict(X_test)

# =====================================
# Evaluation
# =====================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# =====================================
# Confusion Matrix
# =====================================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")

plt.show()

print("\nCompleted Successfully!")