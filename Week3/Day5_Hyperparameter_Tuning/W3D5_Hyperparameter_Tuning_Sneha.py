# ============================================================
# Week 3 Day 5
# Hyperparameter Tuning using GridSearchCV & RandomizedSearchCV
# Name: Sneha G R
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# Create Output Folder
# ============================================================

os.makedirs("outputs", exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

dataset = X.copy()
dataset["Target"] = y
dataset.to_csv("sample_data.csv", index=False)

print("Dataset Loaded Successfully")

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
# Grid Search
# ============================================================

grid_parameters = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

grid_search = GridSearchCV(
    SVC(),
    grid_parameters,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print("\nGrid Search Best Parameters")
print(grid_search.best_params_)

# ============================================================
# Random Search
# ============================================================

random_parameters = {
    "C": [0.1, 1, 10, 50, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

random_search = RandomizedSearchCV(
    SVC(),
    random_parameters,
    n_iter=5,
    cv=5,
    random_state=42
)

random_search.fit(X_train, y_train)

print("\nRandom Search Best Parameters")
print(random_search.best_params_)

# ============================================================
# Best Model
# ============================================================

best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print(classification_report(y_test, predictions))

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.title("Best Model Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")

plt.close()

# ============================================================
# Save Results
# ============================================================

with open("outputs/best_model.txt", "w") as file:
    file.write("Week 3 Day 5\n\n")
    file.write(f"Grid Search Best Parameters:\n{grid_search.best_params_}\n\n")
    file.write(f"Random Search Best Parameters:\n{random_search.best_params_}\n\n")
    file.write(f"Accuracy: {accuracy:.4f}")

print("\nResults Saved Successfully!")

# ============================================================
# Practice Function
# ============================================================

def practice():
    print("\nHyperparameter tuning completed successfully!")
    print("GridSearch checks all parameter combinations.")
    print("RandomizedSearch checks a random subset and is faster.")

practice()

print("\nDone! Review with CIA for feedback.")