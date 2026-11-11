# Transaction Fraud Detection System

An event-driven, production-ready machine learning API built to detect fraudulent credit card transactions. 

## Architecture & Tech Stack
* **Machine Learning:** XGBoost (handles severe class imbalance, F1-score: 0.83).
* **Backend:** FastAPI (RESTful routing and Pydantic validation).
* **Database:** PostgreSQL (SQLAlchemy ORM for logging transactions and prediction history).
* **Infrastructure:** Docker (Containerized for cross-platform deployment).

## Model Performance Upgrade
Initially built with Logistic Regression, the model was upgraded to an **XGBoost Classifier** using `scale_pos_weight`. 
* Precision: 0.92
* Recall: 0.76
* F1-Score: 0.83
* *False positives dropped significantly, minimizing disruptions for legitimate users.*

## Running Locally (Docker)
Ensure Docker is running, then execute:
```bash
docker build -t transaction-fraud-api .
docker run -p 8000:8000 transaction-fraud-api