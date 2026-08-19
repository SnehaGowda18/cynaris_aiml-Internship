import os
import shap
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

os.makedirs("outputs", exist_ok=True)

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

print(classification_report(y_test, pred))

with open("outputs/fairness_report.txt", "w") as f:
    f.write(classification_report(y_test, pred))

# SHAP Explanation
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

plt.figure(figsize=(10,6))
shap.plots.beeswarm(shap_values, show=False)
plt.savefig("outputs/shap_summary.png", bbox_inches="tight")
plt.close()

print("SHAP summary saved.")