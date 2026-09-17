# W11D2 - MLflow Experiment Tracking and Model Registry

## Description

This project uses MLflow to track five Random Forest hyperparameter experiments on the Iris dataset. The experiments log parameters, accuracy metrics, and trained model artifacts. The best-performing model is registered in the MLflow Model Registry and served using the MLflow model serving API.

## Tools Used

- Python
- Scikit-learn
- MLflow
- Pandas
- Random Forest Classifier
- MLflow Model Registry
- MLflow Model Serving

## Experiments

Five Random Forest models were trained with different hyperparameters:

| Experiment | n_estimators | max_depth | Accuracy |
|------------|--------------|-----------|----------|
| 1 | 50 | 3 | 1.0000 |
| 2 | 100 | 5 | 1.0000 |
| 3 | 150 | 7 | 1.0000 |
| 4 | 200 | 10 | 1.0000 |
| 5 | 250 | None | 1.0000 |

## Model Registry

The best model was registered with the following details:

- Registered Model Name: `W11D2-RandomForest`
- Model Version: `1`
- Best Accuracy: `1.0000`

## Model Serving

The registered model was served using:

```powershell
mlflow models serve -m "models:/W11D2-RandomForest/1" --port 5001 --env-manager local