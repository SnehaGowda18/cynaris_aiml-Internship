import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------
# Create Output Folder
# ---------------------------------

os.makedirs("outputs", exist_ok=True)


# ---------------------------------
# Load Dataset
# ---------------------------------

df = pd.read_csv("data/sample_data.csv")


# ---------------------------------
# Dataset Inspection
# ---------------------------------

print("=" * 50)
print("Dataset Information")
print("=" * 50)

df.info()


print("\nDataset Description")
print("=" * 50)

print(df.describe())


print("\nMissing Values")
print("=" * 50)

print(df.isnull().sum())


# Save EDA output

df.describe(include="all").to_csv(
    "outputs/eda_output.csv"
)


# ---------------------------------
# Distribution Plots
# ---------------------------------

numeric_columns = [
    "Age",
    "Salary",
    "Marks"
]


for col in numeric_columns:

    if col in df.columns:

        plt.figure(figsize=(6,4))

        plt.hist(
            df[col].dropna(),
            bins=10,
            edgecolor="black"
        )

        plt.title(
            f"Distribution of {col}"
        )

        plt.xlabel(col)

        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(
            f"outputs/distribution_{col}.png"
        )

        plt.close()



# ---------------------------------
# Correlation Heatmap
# ---------------------------------

corr_columns = [
    "Age",
    "Height",
    "Weight",
    "Marks",
    "Salary"
]


available_columns = [
    col for col in corr_columns
    if col in df.columns
]


plt.figure(figsize=(7,5))


sns.heatmap(
    df[available_columns].corr(),
    annot=True,
    cmap="coolwarm"
)


plt.title(
    "Correlation Heatmap"
)


plt.tight_layout()


plt.savefig(
    "outputs/correlation_heatmap.png"
)


plt.close()



# ---------------------------------
# Top 10 Category Counts
# ---------------------------------

category_columns = [
    "Government",
    "State"
]


for col in category_columns:

    if col in df.columns:

        plt.figure(figsize=(7,4))


        df[col].value_counts().head(10).plot(
            kind="bar"
        )


        plt.title(
            f"Top 10 {col}"
        )


        plt.xlabel(col)

        plt.ylabel("Count")


        plt.tight_layout()


        plt.savefig(
            f"outputs/top10_{col}.png"
        )


        plt.close()



# ---------------------------------
# EDA Observations
# ---------------------------------

print("\nEDA Observations:")
print("1. Dataset contains numerical and categorical features.")
print("2. Missing values were checked and handled.")
print("3. Distribution plots were generated for Age, Salary and Marks.")
print("4. Correlation heatmap shows relationships between numerical features.")
print("5. Category frequency was analyzed for Government and State.")


print("\nEDA Completed Successfully!")
print("Check the outputs folder.")