from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Sentiment Classifier API")

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


class Review(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Sentiment Classifier API Running"}


@app.post("/predict")
def predict(review: Review):

    text = vectorizer.transform([review.text])

    pred = model.predict(text)[0]

    sentiment = "Positive" if pred == 1 else "Negative"

    return {
        "review": review.text,
        "prediction": sentiment,
    }