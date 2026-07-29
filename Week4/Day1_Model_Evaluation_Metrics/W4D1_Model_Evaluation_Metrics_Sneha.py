"""
============================================================
Week 4 Day 1
Model Evaluation Metrics
Author: Sneha G R

Topics Covered:
1. Train-Test Split
2. Precision
3. Recall
4. F1 Score
5. ROC-AUC
6. Confusion Matrix
7. Classification Report
8. K-Fold Cross Validation
9. Stratified K-Fold
10. Learning Curve
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
    learning_curve
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ============================================================
# Create Output Folder
# ============================================================

os.makedirs("outputs", exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("Loading Breast Cancer Dataset...")
print("=" * 60)

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Samples :", X.shape[0])
print("Features:", X.shape[1])

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# Train Logistic Regression Model
# ============================================================

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

# ============================================================
# Prediction
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ============================================================
# Evaluation Metrics
# ============================================================

print("\n" + "=" * 60)
print("Evaluation Metrics")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ============================================================
# K-Fold Cross Validation
# ============================================================

print("\n" + "=" * 60)
print("K-Fold Cross Validation")
print("=" * 60)

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("Scores:", scores)
print("Average Accuracy:", scores.mean())

# ============================================================
# Stratified K-Fold
# ============================================================

print("\n" + "=" * 60)
print("Stratified K-Fold")
print("=" * 60)

skfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=skfold,
    scoring="accuracy"
)

print("Scores:", scores)
print("Average Accuracy:", scores.mean())

# ============================================================
# ROC Curve
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7,5))

plt.plot(fpr, tpr, linewidth=2, label="ROC Curve")
plt.plot([0,1],[0,1],'k--')

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/roc_curve.png", dpi=300)

plt.close()

print("\nROC Curve saved successfully!")

# ============================================================
# Learning Curve
# ============================================================

train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1,1.0,5)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(7,5))

plt.plot(
    train_sizes,
    train_mean,
    marker='o',
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    train_sizes,
    test_mean,
    marker='o',
    linewidth=2,
    label="Validation Accuracy"
)

plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig("outputs/learning_curve.png", dpi=300)

plt.close()

print("Learning Curve saved successfully!")

# ============================================================
# Save Results to Text File
# ============================================================

with open("outputs/evaluation_output.txt", "w") as file:

    file.write("MODEL EVALUATION METRICS\n")
    file.write("=========================\n\n")

    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall   : {recall:.4f}\n")
    file.write(f"F1 Score : {f1:.4f}\n")
    file.write(f"ROC AUC  : {auc:.4f}\n\n")

    file.write("KFold Average Accuracy\n")
    file.write(str(scores.mean()))

print("Evaluation results saved!")

# ============================================================
# Overfitting / Underfitting Explanation
# ============================================================

print("\n" + "=" * 60)
print("Learning Curve Analysis")
print("=" * 60)

if abs(train_mean[-1] - test_mean[-1]) < 0.05:
    print("Model is Generalizing Well.")
elif train_mean[-1] > test_mean[-1]:
    print("Possible Overfitting Detected.")
else:
    print("Possible Underfitting Detected.")

print("\nProgram Completed Successfully!")

print("\nGenerated Files:")
print("outputs/")
print("   roc_curve.png")
print("   learning_curve.png")
print("   evaluation_output.txt")