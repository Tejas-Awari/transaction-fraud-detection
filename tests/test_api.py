from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "xgboost"

def test_predict_fraud_valid_input():
    # A dummy transaction payload
    payload = {
        "Time": 0.0,
        "Amount": 150.00,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "is_fraud" in response.json()
    assert "fraud_probability" in response.json()

def test_predict_fraud_invalid_amount():
    # Testing Pydantic validation (Amount cannot be negative)
    payload = {
        "Time": 0.0,
        "Amount": -50.00 
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
