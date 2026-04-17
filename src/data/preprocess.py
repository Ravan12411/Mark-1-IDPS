"""
Mark-1 IDPS — Data Preprocessing Module
Loads CICIDS2017, cleans, encodes, splits, and scales for model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess(data_path="data/raw/cicids2017/cicids2017.csv"):
    print(f"[*] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"[+] Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

    # Drop non-predictive columns
    drop_cols = ["Destination Port", " Destination Port"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Handle label column (may have leading space)
    label_col = " Label" if " Label" in df.columns else "Label"
    X = df.drop(columns=[label_col])
    y = df[label_col].astype("category").cat.codes

    # Remove rare classes (< 2 samples) to allow stratified split
    class_counts = y.value_counts()
    rare = class_counts[class_counts < 2].index
    mask = ~y.isin(rare)
    X, y = X[mask], y[mask]

    # Clean infinities and NaNs
    X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Normalize
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[+] Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"[+] Classes: {y.nunique()}")

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


if __name__ == "__main__":
    preprocess()
