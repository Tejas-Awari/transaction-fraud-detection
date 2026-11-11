# Transaction Fraud Detection API

A machine learning project that detects potentially fraudulent credit card transactions using Logistic Regression and exposes predictions through a FastAPI REST API.

## Features

- Credit card transaction data analysis
- Data cleaning and preprocessing
- Handling imbalanced data using class weights
- Logistic Regression classification
- Threshold-based fraud detection
- FastAPI prediction endpoint
- Saved model, scaler, and threshold using Joblib

## Tech Stack

- Python 3.11
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib

## Model Performance

Final test results:

| Metric | Fraud Class |
|---|---:|
| Precision | 0.61 |
| Recall | 0.79 |
| F1-score | 0.69 |

- True Positives: 75
- False Negatives: 20
- False Positives: 47
- Decision threshold: 0.99

## Project Structure

```text
transaction-fraud-detection/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── models/
│   │   ├── fraud_model.pkl
│   │   ├── scaler.pkl
│   │   └── threshold.pkl
│   ├── api.py
│   ├── preprocess.py
│   └── train.py
├── tests/
├── requirements.txt
└── README.md
```

## Dataset

Credit Card Fraud Detection Dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` and place it inside `data/raw/`.

## Installation

```bash
git clone https://github.com/Tejas-Awari/transaction-fraud-detection.git

cd transaction-fraud-detection

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

## Train the Model

```bash
python src/train.py
```

## Start the API

```bash
python -m uvicorn src.api:app --reload
```

Open Swagger documentation:

http://127.0.0.1:8000/docs

## API Endpoint

`POST /predict`

Accepts transaction features and returns:

- Fraud prediction
- Fraud probability
- Decision threshold

