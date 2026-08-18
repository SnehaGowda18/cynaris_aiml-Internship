# ============================================================
# Week 3 Day 4
# SVM vs KNN Classification
# Name: Sneha G R
# ============================================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import os

# ============================================================
# Create Outputs Folder
# ============================================================

os.makedirs("outputs", exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("\nDataset Loaded Successfully")
print(X.head())

# Save dataset
dataset = X.copy()
dataset["Target"] = y
dataset.to_csv("sample_data.csv", index=False)

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# Support Vector Machine
# ============================================================

svm_model = SVC(kernel="linear", random_state=42)

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("\n========== SVM ==========")
print("Accuracy:", svm_accuracy)

print(classification_report(y_test, svm_predictions))

cm = confusion_matrix(y_test, svm_predictions)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.title("SVM Confusion Matrix")

plt.savefig("outputs/svm_confusion_matrix.png")

plt.close()

# ============================================================
# KNN
# ============================================================

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)

print("\n========== KNN ==========")
print("Accuracy:", knn_accuracy)

print(classification_report(y_test, knn_predictions))

cm = confusion_matrix(y_test, knn_predictions)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.title("KNN Confusion Matrix")

plt.savefig("outputs/knn_confusion_matrix.png")

plt.close()

# ============================================================
# Comparison
# ============================================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(f"SVM Accuracy : {svm_accuracy:.4f}")
print(f"KNN Accuracy : {knn_accuracy:.4f}")

if svm_accuracy > knn_accuracy:
    better = "SVM"
elif knn_accuracy > svm_accuracy:
    better = "KNN"
else:
    better = "Both models performed equally"

print("\nBetter Model:", better)

with open("outputs/comparison.txt", "w") as file:
    file.write("Week 3 Day 4 - Model Comparison\n\n")
    file.write(f"SVM Accuracy : {svm_accuracy:.4f}\n")
    file.write(f"KNN Accuracy : {knn_accuracy:.4f}\n")
    file.write(f"Better Model : {better}\n")

print("\nResults saved successfully.")

# ============================================================
# Playground Practice Function
# ============================================================

def practice():

    print("\nPractice Completed Successfully!")

    print("SVM works well for high-dimensional datasets.")

    print("KNN works well for smaller datasets.")

practice()

print("\nDone! Review with CIA for feedback.")