@'
# W12D4 MLOps Monitoring Strategy

## 1. Metrics to Track

### API Metrics
- Request count
- Response time
- HTTP error rate
- API availability

### ML Metrics
- Prediction distribution
- Prediction latency
- Model accuracy when labelled data is available
- Data drift
- Feature distribution changes

### Infrastructure Metrics
- CPU usage
- Memory usage
- Docker container health
- Container restart count

## 2. Alerts

Alerts should be configured for:

- API availability below 99%
- HTTP 5xx errors above 5%
- Response time above 2 seconds
- Container repeatedly restarting
- Memory or CPU usage remaining above 80%
- Significant data drift
- Model accuracy dropping below the defined threshold

## 3. Retraining Triggers

A model retraining should be considered when:

- Model accuracy drops below the agreed threshold.
- Significant data drift is detected.
- The incoming data distribution changes substantially.
- New labelled training data becomes available.
- Prediction quality degrades consistently over time.

## 4. Monitoring Workflow

```text
ML API
  |
  v
Collect Metrics
  |
  v
Monitor API + Model + Infrastructure
  |
  v
Threshold / Drift Detection
  |
  +---- Normal ----> Continue Monitoring
  |
  +---- Alert -----> Investigate
                         |
                         v
                  Retraining Required?
                         |
                    +----+----+
                    |         |
                   No        Yes
                    |         |
                    v         v
                Monitor    Retrain Model
                              |
                              v
                       Validate & Deploy