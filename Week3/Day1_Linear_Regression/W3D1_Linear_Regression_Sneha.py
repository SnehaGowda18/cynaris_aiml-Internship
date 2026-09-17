# =====================================================
# W3D1: Linear Regression using Scikit-Learn
# Name: Sneha G R
# =====================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =====================================================
# Create Output Folder
# =====================================================

output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("house_prices.csv")

print("\n========== First 5 Rows ==========")
print(df.head())

print("\n========== Dataset Info ==========")
print(df.info())

# =====================================================
# Handle Missing Values
# =====================================================

df = df.dropna()

# =====================================================
# Convert Categorical Columns
# =====================================================

df = pd.get_dummies(df, drop_first=True)

# =====================================================
# Select Features and Target
# =====================================================

# Target column should be 'price'
X = df.drop("price", axis=1)
y = df["price"]

# =====================================================
# Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# Function to Train & Evaluate
# =====================================================

results = []

def evaluate_model(model, model_name):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([
        model_name,
        mse,
        rmse,
        mae,
        r2
    ])

    print(f"\n========== {model_name} ==========")

    print("Intercept:")
    print(model.intercept_)

    print("\nCoefficients:")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    print(coef_df)

    print("\nMSE :", mse)
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)

    return y_pred

# =====================================================
# Linear Regression
# =====================================================

linear = LinearRegression()
linear_predictions = evaluate_model(linear, "Linear Regression")

# =====================================================
# Ridge Regression
# =====================================================

ridge = Ridge(alpha=1.0)
evaluate_model(ridge, "Ridge Regression")

# =====================================================
# Lasso Regression
# =====================================================

lasso = Lasso(alpha=1.0)
evaluate_model(lasso, "Lasso Regression")

# =====================================================
# Predicted vs Actual Plot
# =====================================================

plt.figure(figsize=(7,6))

plt.scatter(y_test, linear_predictions)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual")

plt.tight_layout()

plt.savefig(os.path.join(output_folder, "predicted_vs_actual.png"))

plt.show()

# =====================================================
# Residual Plot
# =====================================================

residuals = y_test - linear_predictions

plt.figure(figsize=(7,6))

plt.scatter(linear_predictions, residuals)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(os.path.join(output_folder, "residual_plot.png"))

plt.show()

# =====================================================
# Comparison Table
# =====================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MSE",
        "RMSE",
        "MAE",
        "R2 Score"
    ]
)

print("\n========== Model Comparison ==========")
print(results_df)

results_df.to_csv(
    os.path.join(output_folder, "model_comparison.csv"),
    index=False
)

print("\nAll outputs saved successfully!")