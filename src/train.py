import joblib
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/raw/creditcard.csv")

# Remove duplicates
df = df.drop_duplicates()

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split 1: Separate final test data
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split 2: Separate training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

# Scale Amount and Time
scaler = StandardScaler()

X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

X_train[["Amount", "Time"]] = scaler.fit_transform(
    X_train[["Amount", "Time"]]
)

X_val[["Amount", "Time"]] = scaler.transform(
    X_val[["Amount", "Time"]]
)

X_test[["Amount", "Time"]] = scaler.transform(
    X_test[["Amount", "Time"]]
)

# Scale numerical columns
scaler = StandardScaler()

X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

# Fit only on training data
X_train[["Amount", "Time"]] = scaler.fit_transform(
    X_train[["Amount", "Time"]]
)

# Transform validation and test data
X_val[["Amount", "Time"]] = scaler.transform(
    X_val[["Amount", "Time"]]
)

X_test[["Amount", "Time"]] = scaler.transform(
    X_test[["Amount", "Time"]]
)
# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_prob = model.predict_proba(X_val)[:, 1]
# Final evaluation on unseen test data
FINAL_THRESHOLD = 0.99

y_test_prob = model.predict_proba(X_test)[:, 1]

y_test_pred = (
    y_test_prob >= FINAL_THRESHOLD
).astype(int)

print("\n--- FINAL TEST RESULTS ---")
print("Selected threshold:", FINAL_THRESHOLD)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Save trained model and scaler
MODEL_DIR = Path("src/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_DIR / "fraud_model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

joblib.dump(
    FINAL_THRESHOLD,
    MODEL_DIR / "threshold.pkl"
)

print("\nModel, scaler and threshold saved successfully!")