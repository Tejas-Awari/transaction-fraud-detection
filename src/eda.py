
import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv")

# 1. Transaction distribution
print("\n--- TRANSACTION DISTRIBUTION ---")
print(df["Class"].value_counts())

# 2. Average transaction amount
print("\n--- AVERAGE TRANSACTION AMOUNT ---")
print(df["Amount"].mean())

# 3. Amount statistics by transaction type
print("\n--- AMOUNT STATISTICS ---")
print(
    df.groupby("Class")["Amount"].agg(
        ["count", "mean", "median", "min", "max"]
    )
)

# 4. Duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(df.duplicated().sum())

