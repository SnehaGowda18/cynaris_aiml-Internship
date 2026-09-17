# Week 2 - Day 2: Feature Scaling & Selection

## Objective

Implement Feature Scaling and Feature Selection techniques using Python and Scikit-learn.

## Concepts Covered

### Encoding Techniques

* LabelEncoder
* OneHotEncoder
* OrdinalEncoder

### Scaling Techniques

* StandardScaler
* MinMaxScaler
* RobustScaler

### Feature Selection

* SelectKBest
* ANOVA F-Test (f_classif)

## Tools & Technologies Used

* Python 3.11.9
* VS Code
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Git
* GitHub

## Project Files

```text
Day2_Feature_Scaling_Selection/
│
├── outputs/
│   ├── before_scaling.png
│   ├── after_standard_scaler.png
│   ├── after_minmax_scaler.png
│   ├── after_robust_scaler.png
│   └── top5_features.csv
│
├── README.md
├── requirements.txt
└── W2D2_Feature_Scaling_Selection_Sneha.py
```

## Tasks Completed

* Applied LabelEncoder on Gender column
* Applied OneHotEncoder on Department column
* Applied OrdinalEncoder on Experience column
* Applied StandardScaler on numerical features
* Applied MinMaxScaler on numerical features
* Applied RobustScaler on numerical features
* Generated plots before and after scaling
* Used SelectKBest to identify top 5 important features
* Saved feature scores to CSV

## Output Files

### Visualizations

* before_scaling.png
* after_standard_scaler.png
* after_minmax_scaler.png
* after_robust_scaler.png

### Feature Selection

* top5_features.csv

## Trade-offs

### LabelEncoder

Advantages:

* Simple and fast
* Works well for binary categories

Limitations:

* Introduces artificial ordering

### OneHotEncoder

Advantages:

* No false ordering
* Suitable for nominal categories

Limitations:

* Increases dimensionality

### OrdinalEncoder

Advantages:

* Preserves category order
* Efficient representation

Limitations:

* Only suitable for ordered categories

## Viva Answers

### When should you use OneHotEncoder vs OrdinalEncoder?

OneHotEncoder should be used for categorical variables with no natural order, such as city names or departments. OrdinalEncoder should be used when categories have a meaningful order, such as Beginner, Intermediate, and Expert.

### Why does StandardScaler not work well with outliers?

StandardScaler uses mean and standard deviation, which are sensitive to extreme values. Outliers can distort the scaling process and affect model performance.

### What is feature leakage and how do you prevent it?

Feature leakage occurs when information from the target variable or future data is unintentionally used during model training. It can be prevented by splitting data before preprocessing and ensuring that only training data is used to fit preprocessing steps.

## Result

Successfully implemented encoding, scaling, visualization, and feature selection techniques and generated the required outputs.
