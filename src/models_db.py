from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from .database import Base

class TransactionRecord(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    amount = Column(Float, nullable=False)
    # Add other key features if you want, or store the prediction result
    is_fraud = Column(Integer, nullable=False)
    fraud_probability = Column(Float, nullable=False)
