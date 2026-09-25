from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "W12D4 ML API is running"


def test_predict():
    response = client.post("/predict", json={"value": 8})
    assert response.status_code == 200
    assert response.json()["prediction"] == "High"
