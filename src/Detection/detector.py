"""
Mark-1 IDPS — Anomaly Detector
Runs trained Random Forest model on unseen network traffic CSV.
Outputs per-row predictions with attack classification.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


LABEL_MAP = {
    0: "BENIGN",
    1: "DoS Hulk",
    2: "PortScan",
    3: "DDoS",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
    6: "SSH-Patator",
    7: "DoS slowloris",
    8: "DoS Slowhttptest",
    9: "Bot",
    10: "Web Attack",
    11: "Infiltration",
    12: "Heartbleed",
}


def load_model(model_path="results/rf_model.pkl"):
    if not os.path.exists(model_path):
        print(f"[-] Model not found at {model_path}")
        print("    Train first: python main.py train --model baseline")
        sys.exit(1)
    return joblib.load(model_path)


def preprocess_input(df):
    """Apply same preprocessing as training pipeline."""
    drop_cols = ["Destination Port", " Destination Port", "Label", " Label"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.fillna(0)
    scaler = MinMaxScaler()
    return scaler.fit_transform(df), df.index


def detect(input_path):
    print(f"[*] Loading input: {input_path}")
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except Exception as e:
        print(f"[-] Failed to read CSV: {e}")
        sys.exit(1)

    print(f"[*] Loaded {len(df)} records.")
    X_scaled, idx = preprocess_input(df)

    model = load_model()
    predictions = model.predict(X_scaled)

    # Build results
    results = pd.DataFrame({
        "row": idx,
        "prediction_code": predictions,
        "label": [LABEL_MAP.get(p, f"Attack-{p}") for p in predictions],
    })

    total = len(results)
    benign = (results["prediction_code"] == 0).sum()
    attacks = total - benign

    print(f"\n===== DETECTION RESULTS =====")
    print(f"  Total records : {total}")
    print(f"  Benign        : {benign} ({benign/total*100:.1f}%)")
    print(f"  Attacks       : {attacks} ({attacks/total*100:.1f}%)")

    if attacks > 0:
        print("\n  Attack breakdown:")
        attack_rows = results[results["prediction_code"] != 0]
        for label, count in attack_rows["label"].value_counts().items():
            print(f"    {label}: {count}")

    # Save report
    os.makedirs("results", exist_ok=True)
    out_path = "results/detection_report.csv"
    results.to_csv(out_path, index=False)
    print(f"\n[+] Full report saved to {out_path}")
