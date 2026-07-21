# ==========================================================
# Week 2 Day 3
# Handling Imbalanced Data using SMOTE
# Author: Sneha G R
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# ==========================================================
# Create outputs folder
# ==========================================================

os.makedirs("outputs", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("data/customer_data.csv")

print("\n========== ORIGINAL DATA ==========\n")
print(df)

# ==========================================================
# Original Class Distribution
# ==========================================================

print("\n========== ORIGINAL CLASS DISTRIBUTION ==========\n")
print(df["Purchased"].value_counts())

# ==========================================================
# Plot Original Distribution
# ==========================================================

plt.figure(figsize=(6,4))

df["Purchased"].value_counts().sort_index().plot(
    kind="bar",
    color=["skyblue","orange"]
)

plt.title("Original Class Distribution")
plt.xlabel("Purchased")
plt.ylabel("Count")
plt.xticks([0,1],["0","1"])

plt.tight_layout()

plt.savefig("outputs/original_distribution.png")

plt.close()

# ==========================================================
# Features and Target
# ==========================================================

X = df[["Age","Salary"]]
y = df["Purchased"]

# ==========================================================
# Apply SMOTE
# ==========================================================
# Since our dataset has only 2 minority samples,
# use k_neighbors=1.

smote = SMOTE(
    random_state=42,
    k_neighbors=1
)

X_resampled, y_resampled = smote.fit_resample(X, y)

# ==========================================================
# Balanced Dataset
# ==========================================================

balanced_df = pd.DataFrame(
    X_resampled,
    columns=["Age","Salary"]
)

balanced_df["Purchased"] = y_resampled

print("\n========== BALANCED DATA ==========\n")
print(balanced_df)

# ==========================================================
# Balanced Distribution
# ==========================================================

print("\n========== BALANCED CLASS DISTRIBUTION ==========\n")
print(balanced_df["Purchased"].value_counts())

# ==========================================================
# Plot Balanced Distribution
# ==========================================================

plt.figure(figsize=(6,4))

balanced_df["Purchased"].value_counts().sort_index().plot(
    kind="bar",
    color=["green","red"]
)

plt.title("SMOTE Balanced Class Distribution")
plt.xlabel("Purchased")
plt.ylabel("Count")
plt.xticks([0,1],["0","1"])

plt.tight_layout()

plt.savefig("outputs/smote_distribution.png")

plt.close()

# ==========================================================
# Save Balanced Dataset
# ==========================================================

balanced_df.to_csv(
    "outputs/balanced_dataset.csv",
    index=False
)

# ==========================================================
# Summary
# ==========================================================

print("\n========== SUMMARY ==========\n")

print("Original Samples :", len(df))
print("Balanced Samples :", len(balanced_df))

print("\nTop 5 Rows of Balanced Dataset:\n")
print(balanced_df.head())

print("\nOutputs Generated Successfully:")

print("✓ original_distribution.png")
print("✓ smote_distribution.png")
print("✓ balanced_dataset.csv")

print("\nWeek 2 Day 3 - Handling Imbalanced Data using SMOTE Completed Successfully!")

print("\nDone! Review with CIA for feedback.")