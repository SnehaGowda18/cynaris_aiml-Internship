import mlflow
import mlflow.sklearn

model_uri = "models:/W11D1-Iris-RandomForest/1"

model = mlflow.sklearn.load_model(model_uri)

print("Model loaded successfully!")
print("Model type:", type(model).__name__)

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("Sample prediction:", prediction)
print("Predicted class:", prediction[0])