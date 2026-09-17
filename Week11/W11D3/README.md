# W11D3 - MLflow Experiment Tracking and Model Registry

## Description

Implemented MLflow experiment tracking for an Iris classification model. Five Random Forest experiments were executed with different hyperparameters. Parameters, accuracy metrics, and model artifacts were logged. The best model was registered in the MLflow Model Registry and served through an MLflow REST API.

## Tools Used

- Python
- Scikit-learn
- MLflow
- Pandas
- NumPy
- Uvicorn
- SQLite
- Random Forest Classifier

## Features

- Tracked five model experiments
- Logged hyperparameters and accuracy
- Saved model artifacts
- Registered the best model
- Served the model using MLflow
- Tested predictions through REST API

## Model Details

Dataset: Iris Dataset

Algorithm: Random Forest Classifier

Metric: Accuracy

Registered Model: W11D3-Iris-RandomForest

Serving Port: 5001