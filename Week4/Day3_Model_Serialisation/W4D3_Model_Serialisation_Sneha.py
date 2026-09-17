"""
===========================================================
Week 4 Day 3
Model Serialisation using Joblib & Pickle

Author: Sneha G R
===========================================================
"""

import os
import joblib
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ==========================================================
# Create Folders
# ==========================================================

os.makedirs("saved_models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Breast Cancer Dataset")
print("=" * 60)

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Samples :", X.shape[0])
print("Features:", X.shape[1])

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# Train Model
# ==========================================================

print("\nTraining Logistic Regression Model...")

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# ==========================================================
# Predictions
# ==========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy, 4))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================================
# Save Model using Joblib
# ==========================================================

joblib.dump(model, "saved_models/logistic_model.joblib")

print("\nModel saved using Joblib.")

# ==========================================================
# Save Model using Pickle
# ==========================================================

with open("saved_models/logistic_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved using Pickle.")

# ==========================================================
# Load Joblib Model
# ==========================================================

joblib_model = joblib.load("saved_models/logistic_model.joblib")

print("\nJoblib Model Loaded Successfully.")

# ==========================================================
# Load Pickle Model
# ==========================================================

with open("saved_models/logistic_model.pkl", "rb") as file:
    pickle_model = pickle.load(file)

print("Pickle Model Loaded Successfully.")

# ==========================================================
# Verify Predictions
# ==========================================================

original_prediction = model.predict(X_test)
joblib_prediction = joblib_model.predict(X_test)
pickle_prediction = pickle_model.predict(X_test)

same_joblib = (original_prediction == joblib_prediction).all()
same_pickle = (original_prediction == pickle_prediction).all()

print("\nPrediction Verification")

print("Original == Joblib :", same_joblib)
print("Original == Pickle :", same_pickle)

# ==========================================================
# Save Results
# ==========================================================

with open("outputs/prediction_results.txt", "w") as file:

    file.write("MODEL SERIALISATION RESULTS\n")
    file.write("============================\n\n")

    file.write(f"Accuracy : {accuracy:.4f}\n\n")

    file.write("Original == Joblib : ")
    file.write(str(same_joblib))
    file.write("\n")

    file.write("Original == Pickle : ")
    file.write(str(same_pickle))
    file.write("\n\n")

    file.write("Sample Predictions\n")
    file.write("==================\n")

    for i in range(10):
        file.write(
            f"Sample {i+1}: "
            f"Actual={y_test[i]} "
            f"Prediction={original_prediction[i]}\n"
        )

print("\nPrediction results saved successfully!")

print("\nSaved Files:")
print("saved_models/logistic_model.joblib")
print("saved_models/logistic_model.pkl")
print("outputs/prediction_results.txt")

print("\nProgram Completed Successfully!")