# W12D5 Production ML API Monitoring Strategy

## 1. Metrics to Track

### Application Metrics
- API request count
- Request latency
- HTTP error rate
- HTTP status codes
- Container uptime

### ML Metrics
- Prediction distribution
- Input feature distribution
- Data drift
- Model accuracy when labelled data becomes available

### Infrastructure Metrics
- CPU usage
- Memory usage
- Docker container health
- Container restart count

## 2. Alerts

| Metric | Alert Trigger |
|---|---|
| API error rate | More than 5% errors |
| Response latency | Above 1 second |
| Container status | Container stops or repeatedly restarts |
| Memory usage | Above 80% |
| Data drift | Significant change in input distribution |
| Model accuracy | Accuracy drops below the accepted threshold |

## 3. Retraining Triggers

The model should be considered for retraining when:

1. Data drift remains significant over multiple monitoring periods.
2. Model accuracy drops below the defined production threshold.
3. New labelled training data becomes available.
4. Prediction performance shows a sustained degradation.
5. The data distribution changes significantly from the training dataset.

## 4. Monitoring Workflow

```text
ML API
  |
  v
Collect Metrics
  |
  +---- Application Metrics
  |
  +---- Infrastructure Metrics
  |
  +---- ML/Data Metrics
  |
  v
Check Thresholds
  |
  +---- Normal --> Continue Monitoring
  |
  +---- Alert --> Investigate
                    |
                    v
              Check Model/Data
                    |
                    v
              Retrain if Required