import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MLflow experiment
mlflow.set_experiment("W11D1-Iris-Experiment")

# Five different hyperparameter combinations
experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 5},
    {"n_estimators": 200, "max_depth": None},
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

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model artifact
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            model,
            name="iris_model",
            signature=signature
        )

        print(
            f"Run ID: {run.info.run_id} | "
            f"Params: {params} | "
            f"Accuracy: {accuracy:.4f}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = run.info.run_id

print("\nBest Model")
print("Run ID:", best_run_id)
print("Accuracy:", best_accuracy)

# Register best model
model_uri = f"runs:/{best_run_id}/iris_model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="W11D1-Iris-RandomForest"
)

print("\nRegistered Model:")
print("Name:", registered_model.name)
print("Version:", registered_model.version)