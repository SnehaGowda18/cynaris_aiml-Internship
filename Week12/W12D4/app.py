from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="W12D4 ML API")


class PredictionInput(BaseModel):
    value: float


@app.get("/")
def home():
    return {"message": "W12D4 ML API is running"}


@app.post("/predict")
def predict(data: PredictionInput):
    prediction = "High" if data.value >= 5 else "Low"
    return {
        "input": data.value,
        "prediction": prediction
    }
