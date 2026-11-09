
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/raw/creditcard.csv")

# 1. Remove duplicate records
print("Original shape:", df.shape)

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# 2. Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Scale Amount and Time
scaler = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train[["Amount", "Time"]] = scaler.fit_transform(
    X_train[["Amount", "Time"]]
)

X_test[["Amount", "Time"]] = scaler.transform(
    X_test[["Amount", "Time"]]
)

# 5. Display results
print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining fraud distribution:")
print(y_train.value_counts())

print("\nTesting fraud distribution:")
print(y_test.value_counts())

print("\nPreprocessing completed!")

