# Model Card

## Model Details

- Model: Logistic Regression
- Framework: Scikit-learn

## Intended Use

Binary classification demonstration.

## Training Data

Breast Cancer Wisconsin Dataset.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

## Ethical Considerations

The model should not be used for medical diagnosis without expert review.

## Limitations

- Small dataset
- No demographic fairness evaluation
- Educational purpose only


## Fairness Evaluation

The model was evaluated using IBM AI Fairness 360 on the UCI Adult Income dataset.

Fairness Metric:
- Disparate Impact

Results:
- Before Reweighing: 0.363
- After Reweighing: 1.000

Reweighing successfully reduced demographic bias and improved the fairness of the dataset.