# Production Monitoring Strategy

## 1. Metrics to Monitor

- API availability and health status
- Request latency
- HTTP error rate
- Prediction failures
- Model accuracy
- Input data drift
- Prediction distribution

## 2. Alerts

Set alerts when:

- API health check fails repeatedly
- Response latency becomes consistently high
- HTTP 5xx errors increase
- Prediction requests fail
- Model accuracy drops below the expected threshold
- Significant input data drift is detected

## 3. Retraining Triggers

Retrain the model when:

- Model accuracy drops below the agreed threshold
- Data drift remains significant over time
- New labelled training data becomes available
- Prediction quality degrades consistently

## 4. Monitoring Approach

The Dockerised FastAPI service exposes a /health endpoint for availability checks.

Production monitoring can collect API logs, response latency, error rates, prediction results, and model performance metrics.

MLflow can be used to track model versions, experiments, metrics, and retraining results.

## 5. Recovery

If the production model becomes unreliable:

1. Check application and model logs.
2. Validate incoming data.
3. Compare current model metrics with the previous version.
4. Retrain and evaluate a new model if required.
5. Register the validated model version in MLflow.
6. Deploy the validated version after testing.
