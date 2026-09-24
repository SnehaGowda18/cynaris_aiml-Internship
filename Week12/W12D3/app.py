from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

app = FastAPI(title="W12D3 ML Monitoring API")

iris = load_iris()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(iris.data, iris.target)


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"message": "W12D3 ML Monitoring API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "RandomForest"
    }


@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "model": "RandomForest"
    }
