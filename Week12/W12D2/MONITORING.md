# W12D2 ML API Monitoring Strategy

## 1. Metrics to Track

- API request count
- API response latency
- HTTP error rate
- Container CPU usage
- Container memory usage
- Prediction request failures
- Model prediction distribution
- Model accuracy when labelled production data becomes available

## 2. Alerts

Alerts should be configured for:

- High API latency
- Increased HTTP 4xx/5xx errors
- Container CPU or memory usage above the defined limit
- Repeated prediction failures
- Significant change in prediction distribution
- Model performance dropping below the required accuracy threshold

## 3. Retraining Triggers

The model should be considered for retraining when:

- Model accuracy falls below the agreed threshold.
- Input data distribution shows significant drift.
- Prediction distribution changes significantly.
- New labelled training data becomes available.
- Business or data requirements change.

## 4. Health Monitoring

The `/` endpoint can be used as a basic health check to verify that the API is running.

Docker container status should also be monitored to detect stopped or unhealthy containers.

## 5. Monitoring Workflow

```text
API Metrics
    ↓
Monitor latency, errors and resources
    ↓
Detect data/model drift
    ↓
Evaluate model performance
    ↓
Retrain when thresholds are exceeded
    ↓
Test and deploy updated model