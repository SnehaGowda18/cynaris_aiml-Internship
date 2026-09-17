import mlflow
import mlflow.sklearn

MODEL_URI = r"file:///C:/Users/USER/AppData/Local/Temp/tmpk2jtu6ah/model"

model = mlflow.sklearn.load_model(MODEL_URI)

sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("Sample input:", sample)
print("Predicted class:", prediction[0])