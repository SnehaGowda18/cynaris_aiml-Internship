from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="W12D2 ML API")

iris = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)


class PredictionInput(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "W12D2 ML API is running"}


@app.post("/predict")
def predict(data: PredictionInput):
    prediction = model.predict([data.features])[0]
    return {
        "prediction": int(prediction),
        "class_name": iris.target_names[prediction]
    }