import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif

# =====================================================
# Create Output Folder
# =====================================================

os.makedirs("outputs", exist_ok=True)

# =====================================================
# Sample Dataset
# =====================================================

df = pd.DataFrame({
    "Gender": ["Male", "Female", "Male", "Female", "Male",
               "Female", "Male", "Female", "Male", "Female"],

    "Department": ["IT", "HR", "Finance", "IT", "Sales",
                   "Finance", "HR", "Sales", "IT", "Finance"],

    "Experience": ["Beginner", "Intermediate", "Expert", "Intermediate",
                   "Beginner", "Expert", "Intermediate",
                   "Beginner", "Expert", "Intermediate"],

    "Age": [22, 26, 35, 30, 24, 41, 33, 29, 38, 31],

    "Salary": [30000, 42000, 75000, 50000, 35000,
               90000, 65000, 48000, 85000, 52000],

    "Performance": [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
})

print("\n========== ORIGINAL DATA ==========\n")
print(df)

# =====================================================
# Label Encoding
# =====================================================

print("\n========== LABEL ENCODER ==========\n")

label = LabelEncoder()

df["Gender_Label"] = label.fit_transform(df["Gender"])

print(df[["Gender", "Gender_Label"]])

# =====================================================
# OneHotEncoder
# =====================================================

print("\n========== ONE HOT ENCODER ==========\n")

ct = ColumnTransformer(
    transformers=[
        ("department", OneHotEncoder(sparse_output=False), ["Department"])
    ],
    remainder="passthrough"
)

encoded = ct.fit_transform(df)

encoded_columns = (
    ct.named_transformers_["department"]
      .get_feature_names_out(["Department"])
)

remaining_columns = [
    "Gender",
    "Experience",
    "Age",
    "Salary",
    "Performance",
    "Gender_Label"
]

encoded_df = pd.DataFrame(
    encoded,
    columns=list(encoded_columns) + remaining_columns
)

print(encoded_df)

# =====================================================
# Ordinal Encoding
# =====================================================

print("\n========== ORDINAL ENCODER ==========\n")

ordinal = OrdinalEncoder(
    categories=[["Beginner", "Intermediate", "Expert"]]
)

df["Experience_Ordinal"] = ordinal.fit_transform(
    df[["Experience"]]
)

print(df[["Experience", "Experience_Ordinal"]])

# =====================================================
# Scaling
# =====================================================

numeric = df[["Age", "Salary"]]

# Before Scaling

numeric.hist(figsize=(8,4))
plt.suptitle("Before Scaling")
plt.tight_layout()
plt.savefig("outputs/before_scaling.png")
plt.close()

# StandardScaler

standard = StandardScaler()

standard_df = pd.DataFrame(
    standard.fit_transform(numeric),
    columns=numeric.columns
)

standard_df.hist(figsize=(8,4))
plt.suptitle("After StandardScaler")
plt.tight_layout()
plt.savefig("outputs/after_standard_scaler.png")
plt.close()

# MinMaxScaler

minmax = MinMaxScaler()

minmax_df = pd.DataFrame(
    minmax.fit_transform(numeric),
    columns=numeric.columns
)

minmax_df.hist(figsize=(8,4))
plt.suptitle("After MinMaxScaler")
plt.tight_layout()
plt.savefig("outputs/after_minmax_scaler.png")
plt.close()

# RobustScaler

robust = RobustScaler()

robust_df = pd.DataFrame(
    robust.fit_transform(numeric),
    columns=numeric.columns
)

robust_df.hist(figsize=(8,4))
plt.suptitle("After RobustScaler")
plt.tight_layout()
plt.savefig("outputs/after_robust_scaler.png")
plt.close()

print("\nScaling Completed Successfully.")

# =====================================================
# Feature Selection
# =====================================================

print("\n========== SELECT KBEST ==========\n")

feature_df = pd.DataFrame()

feature_df["Age"] = df["Age"]
feature_df["Salary"] = df["Salary"]
feature_df["Gender"] = df["Gender_Label"]
feature_df["Experience"] = df["Experience_Ordinal"]

department_features = pd.DataFrame(
    OneHotEncoder(sparse_output=False)
    .fit_transform(df[["Department"]]),
    columns=OneHotEncoder()
    .fit(df[["Department"]])
    .get_feature_names_out(["Department"])
)

feature_df = pd.concat(
    [feature_df, department_features],
    axis=1
)

X = feature_df
y = df["Performance"]

selector = SelectKBest(
    score_func=f_classif,
    k=5
)

selector.fit(X, y)

selected = X.columns[selector.get_support()]

print("Top 5 Features:\n")

for feature in selected:
    print(feature)

scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
})

scores.to_csv(
    "outputs/top5_features.csv",
    index=False
)

print("\n========== WHY THESE FEATURES MATTER ==========\n")

reasons = {
    "Age":
        "Employee age may influence work experience.",
    "Salary":
        "Salary reflects employee level and responsibility.",
    "Gender":
        "Demographic feature.",
    "Experience":
        "Experience level impacts performance."
}

for feature in selected:

    if feature in reasons:
        print(f"{feature}: {reasons[feature]}")
    else:
        print(f"{feature}: Department information helps distinguish employee roles.")

# =====================================================
# Practice Function
# =====================================================

def practice():

    print("\nFeature Scaling & Selection Completed Successfully!")

    print("Three Scaling Techniques Applied")

    print("Three Encoding Techniques Applied")

    print("Top Features Selected Successfully")

practice()

print("\nDone! Review with CIA for feedback.")