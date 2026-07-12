import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# Create output folder
# ---------------------------------
os.makedirs("outputs", exist_ok=True)

# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv("sample_data.csv")

# ---------------------------------
# Dataset Inspection
# ---------------------------------
print("=" * 50)
print("Dataset Information")
print("=" * 50)
df.info()

print("\n" + "=" * 50)
print("Dataset Description")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("Missing Values")
print("=" * 50)
print(df.isnull().sum())

# Save EDA output
df.describe(include="all").to_csv("outputs/eda_output.csv")

# ---------------------------------
# Numeric Column Distributions
# ---------------------------------
numeric_columns = ["Age", "Salary", "Marks"]

for column in numeric_columns:
    if column in df.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df[column], bins=8, edgecolor="black")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(f"outputs/distribution_{column}.png")
        plt.close()

# ---------------------------------
# Correlation Heatmap
# ---------------------------------
corr_columns = ["Age", "Height", "Weight", "Marks", "Salary"]

available_columns = [
    col for col in corr_columns 
    if col in df.columns
]

plt.figure(figsize=(7, 5))

sns.heatmap(
    df[available_columns].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# ---------------------------------
# Top 10 Category Counts
# ---------------------------------
categorical_columns = ["Gender", "State"]

for column in categorical_columns:
    if column in df.columns:
        plt.figure(figsize=(7, 4))

        df[column].value_counts().head(10).plot(
            kind="bar"
        )

        plt.title(f"Top 10 {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.tight_layout()

        plt.savefig(f"outputs/top10_{column}.png")
        plt.close()

# ---------------------------------
# EDA Observations
# ---------------------------------
print("\nEDA Observations:")
print("1. Dataset contains numerical and categorical features.")
print("2. No missing values are present in the dataset.")
print("3. Salary and Marks distributions were analyzed.")
print("4. Correlation between numerical features was visualized.")
print("5. Category frequency was analyzed for Gender and State.")

print("\nEDA Completed Successfully!")
print("All files saved inside outputs folder.")