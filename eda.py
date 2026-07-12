import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/sample_data.csv")

print("=" * 50)
print("Dataset Info")
print("=" * 50)
print(df.info())

print("\nDescribe")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# Save describe output
df.describe(include="all").to_csv("outputs/eda_output.csv")

# Numeric Columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Distribution plots
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col].dropna(), bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"outputs/distribution_{col}.png")
    plt.close()

# Correlation Heatmap
if len(numeric_cols) > 1:
    plt.figure(figsize=(8,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png")
    plt.close()

# Categorical Columns
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    plt.figure(figsize=(8,4))
    df[col].value_counts().head(10).plot(kind="bar")
    plt.title(f"Top 10 {col}")
    plt.tight_layout()
    plt.savefig(f"outputs/top10_{col}.png")
    plt.close()

print("\nEDA Completed Successfully")