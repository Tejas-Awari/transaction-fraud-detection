import joblib
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from sqlalchemy.orm import Session

from .database import engine, Base, get_db
from .models_db import TransactionRecord

# Create database tables if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Transaction Fraud Detection API")

# Load model, scaler, and threshold
MODEL_DIR = Path("src/models")
model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
threshold = joblib.load(MODEL_DIR / "threshold.pkl")

# Pydantic input validation model
class TransactionInput(BaseModel):
    Time: float
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = Field(..., ge=0, description="Transaction amount must be non-negative")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "xgboost"}

@app.post("/predict")
def predict_fraud(transaction: TransactionInput, db: Session = Depends(get_db)):
    # Convert input to DataFrame
    input_data = pd.DataFrame([transaction.dict()])
    
    # Scale Amount and Time using the loaded scaler
    input_data[["Amount", "Time"]] = scaler.transform(input_data[["Amount", "Time"]])
    
    # Predict probability
    prob = float(model.predict_proba(input_data)[:, 1][0])
    is_fraud = int(prob >= threshold)
    
    # Save transaction and prediction to PostgreSQL
    db_record = TransactionRecord(
        amount=transaction.Amount,
        is_fraud=is_fraud,
        fraud_probability=prob
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return {
        "fraud_probability": prob,
        "is_fraud": bool(is_fraud),
        "threshold_used": threshold,
        "database_record_id": db_record.id
    }