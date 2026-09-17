import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.feature_selection import SelectKBest, f_classif

# =====================================================
# Create outputs folder
# =====================================================

os.makedirs("outputs", exist_ok=True)

# =====================================================
# Sample Dataset
# =====================================================

df = pd.DataFrame({
    "Gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male"],
    "City": ["Bangalore", "Mysore", "Delhi", "Delhi", "Mysore", "Bangalore", "Chennai", "Delhi"],
    "Education": ["High", "Medium", "Low", "Medium", "High", "Low", "Medium", "High"],
    "Age": [22, 28, 35, 40, 30, 25, 27, 45],
    "Salary": [25000, 40000, 60000, 80000, 50000, 30000, 45000, 90000],
    "Purchased": [0, 1, 1, 1, 0, 0, 1, 1]
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
# One Hot Encoding
# =====================================================

print("\n========== ONE HOT ENCODER ==========\n")

onehot_df = pd.get_dummies(df, columns=["City"], dtype=int)

print(onehot_df)

# =====================================================
# Ordinal Encoding
# =====================================================

print("\n========== ORDINAL ENCODER ==========\n")

ordinal = OrdinalEncoder(categories=[["Low", "Medium", "High"]])

df["Education_Ordinal"] = ordinal.fit_transform(df[["Education"]])

print(df[["Education", "Education_Ordinal"]])

# =====================================================
# Scaling
# =====================================================

features = df[["Age", "Salary"]]

# Before Scaling
features.hist(figsize=(8,4))
plt.suptitle("Before Scaling")
plt.tight_layout()
plt.savefig("outputs/before_scaling.png")
plt.close()

# Standard Scaler
standard = StandardScaler()
standard_scaled = pd.DataFrame(
    standard.fit_transform(features),
    columns=features.columns
)

standard_scaled.hist(figsize=(8,4))
plt.suptitle("After StandardScaler")
plt.tight_layout()
plt.savefig("outputs/after_standard_scaler.png")
plt.close()

# MinMax Scaler
minmax = MinMaxScaler()
minmax_scaled = pd.DataFrame(
    minmax.fit_transform(features),
    columns=features.columns
)

minmax_scaled.hist(figsize=(8,4))
plt.suptitle("After MinMaxScaler")
plt.tight_layout()
plt.savefig("outputs/after_minmax_scaler.png")
plt.close()

# Robust Scaler
robust = RobustScaler()
robust_scaled = pd.DataFrame(
    robust.fit_transform(features),
    columns=features.columns
)

robust_scaled.hist(figsize=(8,4))
plt.suptitle("After RobustScaler")
plt.tight_layout()
plt.savefig("outputs/after_robust_scaler.png")
plt.close()

print("\nScaling Completed.")

# =====================================================
# Feature Selection
# =====================================================

print("\n========== SELECT KBEST ==========\n")

X = pd.DataFrame()

X["Age"] = df["Age"]
X["Salary"] = df["Salary"]
X["Gender_Label"] = df["Gender_Label"]
X["Education_Ordinal"] = df["Education_Ordinal"]

city_encoded = pd.get_dummies(df["City"], prefix="City", dtype=int)

X = pd.concat([X, city_encoded], axis=1)

y = df["Purchased"]

selector = SelectKBest(score_func=f_classif, k=5)

selector.fit(X, y)

selected_features = X.columns[selector.get_support()]

print("Top 5 Features:\n")

for feature in selected_features:
    print(feature)

pd.DataFrame({
    "Top Features": selected_features
}).to_csv("outputs/top_features.csv", index=False)

# =====================================================
# Why these features matter
# =====================================================

print("\n========== WHY THESE FEATURES MATTER ==========\n")

reasons = {
    "Age": "Customer age influences purchasing behaviour.",
    "Salary": "Income affects buying capacity.",
    "Gender_Label": "Represents customer demographic.",
    "Education_Ordinal": "Education level may influence purchasing decisions."
}

for feature in selected_features:
    print(f"{feature} : {reasons.get(feature, 'Represents customer location information.')}")

# =====================================================
# Practice Function
# =====================================================

def practice():
    print("\nFeature Engineering & Encoding Completed Successfully!")

practice()

print("\nDone! Review with CIA for feedback.")