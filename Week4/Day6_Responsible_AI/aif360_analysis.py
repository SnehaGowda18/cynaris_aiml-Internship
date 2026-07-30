from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# Load Adult Income Dataset
dataset = AdultDataset()

print("=" * 50)
print("Adult Dataset Loaded Successfully")
print("=" * 50)

# Fairness Metric Before Reweighing
metric = BinaryLabelDatasetMetric(
    dataset,
    unprivileged_groups=[{"sex": 0}],
    privileged_groups=[{"sex": 1}]
)

before = metric.disparate_impact()

print(f"\nDisparate Impact (Before Reweighing): {before:.3f}")

# Apply Reweighing
RW = Reweighing(
    unprivileged_groups=[{"sex": 0}],
    privileged_groups=[{"sex": 1}]
)

dataset_transformed = RW.fit_transform(dataset)

metric_after = BinaryLabelDatasetMetric(
    dataset_transformed,
    unprivileged_groups=[{"sex": 0}],
    privileged_groups=[{"sex": 1}]
)

after = metric_after.disparate_impact()

print(f"Disparate Impact (After Reweighing): {after:.3f}")

print("\nBias Improvement:", after - before)