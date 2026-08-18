"""
===========================================================
Week 4 Day 2
Bias-Variance Tradeoff & Regularisation

Author : Sneha G R
===========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import uniform

from sklearn.datasets import load_diabetes
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# Create Outputs Folder
# ============================================================

os.makedirs("outputs", exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("Loading Diabetes Dataset")
print("=" * 60)

data = load_diabetes()

X = data.data
y = data.target

print("Samples :", X.shape[0])
print("Features:", X.shape[1])

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
# Ridge Regression
# ============================================================

print("\nTraining Ridge Regression...")

ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print("Ridge MSE :", ridge_mse)
print("Ridge R2  :", ridge_r2)

# ============================================================
# Lasso Regression
# ============================================================

print("\nTraining Lasso Regression...")

lasso = Lasso(alpha=0.1)

lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)

lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print("Lasso MSE :", lasso_mse)
print("Lasso R2  :", lasso_r2)

# ============================================================
# Grid Search
# ============================================================

print("\nRunning GridSearchCV...")

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring="r2"
)

grid.fit(X_train, y_train)

print("Best Alpha :", grid.best_params_)
print("Best Score :", grid.best_score_)

with open("outputs/grid_search_results.txt", "w") as file:
    file.write("Grid Search Results\n")
    file.write("====================\n")
    file.write(str(grid.best_params_))
    file.write("\n")
    file.write(str(grid.best_score_))

# ============================================================
# Randomized Search
# ============================================================

print("\nRunning RandomizedSearchCV...")

param_dist = {
    "alpha": uniform(0.001, 100)
}

random = RandomizedSearchCV(
    Ridge(),
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    random_state=42,
    scoring="r2"
)

random.fit(X_train, y_train)

print("Best Alpha :", random.best_params_)
print("Best Score :", random.best_score_)

with open("outputs/random_search_results.txt", "w") as file:
    file.write("Random Search Results\n")
    file.write("======================\n")
    file.write(str(random.best_params_))
    file.write("\n")
    file.write(str(random.best_score_))

# ============================================================
# Plot Comparison
# ============================================================

models = ["Ridge", "Lasso"]
scores = [ridge_r2, lasso_r2]

plt.figure(figsize=(6,5))
plt.bar(models, scores)

plt.title("Regularisation Comparison")
plt.xlabel("Models")
plt.ylabel("R² Score")

plt.tight_layout()

plt.savefig("outputs/regularization_comparison.png", dpi=300)

plt.close()

print("Comparison graph saved.")

# ============================================================
# Save Best Model
# ============================================================

best_model = "Ridge"

if lasso_r2 > ridge_r2:
    best_model = "Lasso"

with open("outputs/best_model_metrics.txt", "w") as file:

    file.write("Best Model\n")
    file.write("===================\n")
    file.write(f"Best Model : {best_model}\n\n")

    file.write(f"Ridge MSE : {ridge_mse:.2f}\n")
    file.write(f"Ridge R2  : {ridge_r2:.4f}\n\n")

    file.write(f"Lasso MSE : {lasso_mse:.2f}\n")
    file.write(f"Lasso R2  : {lasso_r2:.4f}\n")

print("\nResults saved successfully.")

# ============================================================
# Bias Variance Explanation
# ============================================================

print("\n" + "=" * 60)
print("Bias-Variance Tradeoff")
print("=" * 60)

if ridge_r2 > lasso_r2:
    print("Ridge performed better.")
    print("L2 Regularisation helps reduce variance.")
else:
    print("Lasso performed better.")
    print("L1 Regularisation performs feature selection.")

print("\nProgram Completed Successfully!")