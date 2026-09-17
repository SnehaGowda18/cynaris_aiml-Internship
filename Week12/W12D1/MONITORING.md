# ML API Monitoring Strategy

## Metrics to Track

- API response time
- Request count
- Error rate
- Prediction distribution
- Model accuracy
- Data drift

## Alerts

- Response time above 2 seconds
- Error rate above 5%
- Data drift detected
- Model accuracy below 90%

## Retraining Triggers

- Model accuracy drops below 90%
- Significant data drift
- New labeled training data becomes available
- Increase in incorrect predictions

## Monitoring Tools

- Docker logs
- GitHub Actions
- MLflow
- Python logging