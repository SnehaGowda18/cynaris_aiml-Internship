import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Set experiment
mlflow.set_experiment("W11D2-Model-Registry")

experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 7},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 250, "max_depth": None},
]

best_accuracy = 0
best_run_id = None

for params in experiments:

    with mlflow.start_run() as run:

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            name="random_forest_model"
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Parameters: {params}")
        print(f"Accuracy: {accuracy:.4f}")
        print("-" * 40)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = run.info.run_id

print("Best Model Run ID:", best_run_id)
print("Best Accuracy:", best_accuracy)

# Register the best model
model_uri = f"runs:/{best_run_id}/random_forest_model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="W11D2-RandomForest"
)

print("Registered Model Name:", registered_model.name)
print("Registered Model Version:", registered_model.version)