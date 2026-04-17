"""
Mark-1 IDPS — Model Evaluation Module
Loads trained models and prints full classification metrics.
"""

import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


def evaluate_all(X_test, y_test):
    """
    Evaluate all available trained models against test data.
    """
    results = {}

    # --- Baseline: Random Forest ---
    rf_path = "results/rf_model.pkl"
    if os.path.exists(rf_path):
        import joblib
        print("\n[*] Evaluating Random Forest...")
        rf = joblib.load(rf_path)
        y_pred_rf = rf.predict(X_test)
        results["Random Forest"] = _compute_metrics(y_test, y_pred_rf)
        print(classification_report(y_test, y_pred_rf))
    else:
        print("[-] Random Forest model not found. Run: python main.py train --model baseline")

    # --- Deep: Autoencoder ---
    ae_path = "results/autoencoder_model.h5"
    threshold_path = "results/threshold.npy"
    if os.path.exists(ae_path) and os.path.exists(threshold_path):
        try:
            from tensorflow import keras
            print("\n[*] Evaluating Autoencoder...")
            model = keras.models.load_model(ae_path)
            threshold = np.load(threshold_path)
            reconstructions = model.predict(X_test)
            mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
            y_pred_ae = (mse > threshold).astype(int)
            y_binary = (y_test != 0).astype(int)
            results["Autoencoder"] = _compute_metrics(y_binary, y_pred_ae)
            print(classification_report(y_binary, y_pred_ae, target_names=["Normal", "Attack"]))
        except ImportError:
            print("[-] TensorFlow not installed. Skipping autoencoder evaluation.")
    else:
        print("[-] Autoencoder model not found. Run: python main.py train --model deep")

    # --- Summary ---
    if results:
        print("\n===== EVALUATION SUMMARY =====")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

    return results


def _compute_metrics(y_true, y_pred):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
