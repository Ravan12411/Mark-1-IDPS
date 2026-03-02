import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/cicids2017/cicids2017.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Inspect the first few rows to see column names and values
print(df.head())