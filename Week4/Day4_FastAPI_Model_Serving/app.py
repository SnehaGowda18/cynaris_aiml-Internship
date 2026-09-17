"""
W4D4 - FastAPI Model Serving Endpoint
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI(
    title="Iris Prediction API",
    description="FastAPI Model Serving Example",
    version="1.0"
)

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "FastAPI Model Serving is Running!"
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    species = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    return {
        "prediction": int(prediction),
        "species": species[prediction]
    }