from pathlib import Path

import pandas as pd


# Locate the dataset relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"

# Read the CSV into a pandas DataFrame.
df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())

print("\nMachine failure counts:")
print(df["Machine failure"].value_counts())

print("\nFailure percentage:")
failure_percentage = df["Machine failure"].mean() * 100
print(f"{failure_percentage:.2f}%")

