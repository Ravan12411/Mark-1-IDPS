import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Print library versions
print(pd.__version__, np.__version__, sklearn.__version__)

# Load dataset
df = pd.read_csv("data/raw/cicids2017/cicids2017.csv")

# Drop non-predictive or redundant columns
drop_cols = ["Destination Port"]  # add more if needed
df = df.drop(columns=drop_cols)

# Separate features and target
X = df.drop(columns=["Label"])
y = df["Label"]

# Encode target labels (BENIGN vs attacks)
y = y.astype("category").cat.codes

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)
print("Class distribution in train:\n", pd.Series(y_train).value_counts())