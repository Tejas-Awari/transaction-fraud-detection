
import pandas as pd

DATA_PATH = "data/raw/creditcard.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic information
print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Transactions:")
print(df.head())

print("\nColumn Information:")
print(df.info())

print("\nFraud Distribution:")
print(df["Class"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nFraud Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)

