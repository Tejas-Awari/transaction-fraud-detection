
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# Load saved model artifacts
MODEL_DIR = Path(__file__).resolve().parent / "models"

model = joblib.load(MODEL_DIR / "fraud_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
threshold = joblib.load(MODEL_DIR / "threshold.pkl")


app = FastAPI(
    title="Transaction Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0.0"
)


# Input schema
class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


@app.get("/")
def home():
    return {
        "message": "Transaction Fraud Detection API is running"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    # Convert input into DataFrame
    data = pd.DataFrame([transaction.model_dump()])

    # Scale Amount and Time
    data[["Amount", "Time"]] = scaler.transform(
        data[["Amount", "Time"]]
    )

    # Predict fraud probability
    fraud_probability = model.predict_proba(data)[0][1]

    # Apply selected threshold
    prediction = int(fraud_probability >= threshold)

    return {
        "is_fraud": bool(prediction),
        "prediction": "Fraud" if prediction else "Legitimate",
        "fraud_probability": round(float(fraud_probability), 8),
        "threshold": threshold
    }

