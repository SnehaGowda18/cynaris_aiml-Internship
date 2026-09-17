import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment_name = "W11D3-Iris-Model-Serving"
model_name = "W11D3-Iris-RandomForest"

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    raise Exception("Experiment not found")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)

if runs.empty:
    raise Exception("No training runs found")

best_run_id = runs.iloc[0]["run_id"]
best_accuracy = runs.iloc[0]["metrics.accuracy"]

model_uri = f"runs:/{best_run_id}/model"

client = MlflowClient()

try:
    client.create_registered_model(model_name)
except Exception:
    pass

version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=best_run_id
)

print("Best Run ID:", best_run_id)
print("Model Accuracy:", best_accuracy)
print("Registered Model:", model_name)
print("Model Version:", version.version)
print("Model URI:", model_uri)