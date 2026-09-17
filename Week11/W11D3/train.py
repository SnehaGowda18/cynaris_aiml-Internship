import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("W11D3-Iris-Model-Serving")

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42
)

experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 300, "max_depth": 15},
]

for params in experiments:

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_params(params)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print(
            f"Parameters: {params} | "
            f"Accuracy: {accuracy:.4f}"
        )