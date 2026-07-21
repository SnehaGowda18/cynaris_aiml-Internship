# ==========================================================
# Week 2 Day 5
# End-to-End Preprocessing Pipeline
# ==========================================================

import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==========================================================
# Step 1: Create Sample Dataset
# ==========================================================

data = {
    "Age": [22, 25, None, 30, 28],
    "Salary": [30000, 50000, 40000, None, 45000],
    "Gender": ["Male", "Female", "Female", "Male", None],
    "Department": ["HR", "IT", "Finance", "IT", "HR"]
}

df = pd.DataFrame(data)

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(df)

# ==========================================================
# Step 2: Identify Numerical & Categorical Columns
# ==========================================================

numerical_features = ["Age", "Salary"]
categorical_features = ["Gender", "Department"]

# ==========================================================
# Step 3: Numerical Pipeline
# ==========================================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# ==========================================================
# Step 4: Categorical Pipeline
# ==========================================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# ==========================================================
# Step 5: Combine Pipelines
# ==========================================================

preprocessor = ColumnTransformer([
    ("Numerical", numerical_pipeline, numerical_features),
    ("Categorical", categorical_pipeline, categorical_features)
])

# ==========================================================
# Step 6: Apply Pipeline
# ==========================================================

processed_data = preprocessor.fit_transform(df)

print("\n" + "=" * 60)
print("PROCESSED DATA (NUMPY ARRAY)")
print("=" * 60)
print(processed_data)

# ==========================================================
# Step 7: Convert to DataFrame
# ==========================================================

column_names = preprocessor.get_feature_names_out()

processed_df = pd.DataFrame(
    processed_data,
    columns=column_names
)

print("\n" + "=" * 60)
print("PROCESSED DATAFRAME")
print("=" * 60)
print(processed_df)

# ==========================================================
# Step 8: Save CSV
# ==========================================================

current_folder = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(current_folder, "processed_output.csv")

processed_df.to_csv(csv_path, index=False)

print("\nCSV saved successfully!")
print("Location:", csv_path)

# ==========================================================
# Step 9: Test 1 - Original Dataset
# ==========================================================

print("\n" + "=" * 60)
print("TEST 1 : ORIGINAL DATASET")
print("=" * 60)

print(preprocessor.fit_transform(df))

# ==========================================================
# Step 10: Test 2 - Missing Numerical Value
# ==========================================================

print("\n" + "=" * 60)
print("TEST 2 : MISSING AGE")
print("=" * 60)

test2 = df.copy()
test2.loc[0, "Age"] = None

print(preprocessor.fit_transform(test2))

# ==========================================================
# Step 11: Test 3 - New Category
# ==========================================================

print("\n" + "=" * 60)
print("TEST 3 : NEW DEPARTMENT")
print("=" * 60)

test3 = df.copy()
test3.loc[2, "Department"] = "Marketing"

print(preprocessor.fit_transform(test3))

# ==========================================================
# Step 12: Verify Output
# ==========================================================

print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

print("Processed Data Shape :", processed_df.shape)
print("Missing Values       :", processed_df.isnull().sum().sum())
print("CSV Exists           :", os.path.exists(csv_path))

print("\nAll tests completed successfully!")