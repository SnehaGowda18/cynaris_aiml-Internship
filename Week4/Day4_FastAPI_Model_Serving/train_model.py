"""
W4D4 - FastAPI Model Serving
Train a Machine Learning model and save it.
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully.")
print("Saved as model.pkl")