import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)

# ============================================
# Create outputs folder
# ============================================

os.makedirs("outputs", exist_ok=True)

# ============================================
# Dataset
# ============================================

positive = [
    "I love this product",
    "Amazing experience",
    "Excellent service",
    "Fantastic quality",
    "Very happy with the purchase",
    "Highly recommended",
    "Best product ever",
    "Really satisfied",
    "Wonderful experience",
    "Great support",
    "Very impressive",
    "Worth every penny",
    "Outstanding quality",
    "Five stars",
    "I enjoyed using it",
    "Everything works perfectly",
    "Very reliable",
    "Awesome product",
    "Superb experience",
    "Absolutely loved it",
    "Excellent customer service",
    "Very good",
    "Perfect purchase",
    "Nice packaging",
    "Happy customer"
]

negative = [
    "I hate this product",
    "Worst experience ever",
    "Very bad service",
    "Terrible quality",
    "Waste of money",
    "Completely disappointed",
    "Poor performance",
    "Awful support",
    "Not recommended",
    "Very unhappy",
    "Extremely bad",
    "Never buying again",
    "Product is broken",
    "Very poor quality",
    "Bad experience",
    "Does not work",
    "Cheap material",
    "Really disappointed",
    "Customer service is terrible",
    "Worst purchase",
    "Not worth the money",
    "Poor packaging",
    "Horrible experience",
    "Very frustrating",
    "I regret buying this"
]

texts = (positive * 2) + (negative * 2)
labels = ([1] * len(positive) * 2) + ([0] * len(negative) * 2)

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

# ============================================
# Features
# ============================================

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(df["text"])
y = df["label"]

# ============================================
# Split Data
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ============================================
# Logistic Regression
# ============================================

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

pred_log = log_model.predict(X_test)

print("\n========== LOGISTIC REGRESSION ==========\n")

report = classification_report(y_test, pred_log)

print(report)

with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

# ============================================
# Confusion Matrix
# ============================================

ConfusionMatrixDisplay.from_estimator(
    log_model,
    X_test,
    y_test
)

plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# ============================================
# ROC Curve
# ============================================

RocCurveDisplay.from_estimator(
    log_model,
    X_test,
    y_test
)

plt.title("ROC Curve")
plt.savefig("outputs/roc_curve.png")
plt.close()

# ============================================
# Random Forest
# ============================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

pred_rf = rf_model.predict(X_test)

print("\n========== RANDOM FOREST ==========\n")

print(classification_report(y_test, pred_rf))

# ============================================
# Comparison
# ============================================

print("\n========== MODEL COMPARISON ==========\n")

print(f"Logistic Accuracy  : {accuracy_score(y_test, pred_log):.2f}")
print(f"RandomForest Accuracy : {accuracy_score(y_test, pred_rf):.2f}")

print(f"Logistic Precision : {precision_score(y_test, pred_log):.2f}")
print(f"RandomForest Precision : {precision_score(y_test, pred_rf):.2f}")

print(f"Logistic Recall    : {recall_score(y_test, pred_log):.2f}")
print(f"RandomForest Recall    : {recall_score(y_test, pred_rf):.2f}")

# ============================================
# Save Model
# ============================================

joblib.dump(log_model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully.")
print("Outputs saved inside outputs folder.")