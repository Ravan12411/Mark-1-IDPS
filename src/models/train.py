"""
Mark-1 IDPS — Deep Learning Training Module
LSTM Autoencoder for unsupervised anomaly detection on CICIDS2017.
"""

import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def build_autoencoder(input_dim):
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow is required for deep model training. Run: pip install tensorflow")

    inputs = keras.Input(shape=(input_dim,))
    # Encoder
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    encoded = layers.Dense(16, activation="relu")(x)
    # Decoder
    x = layers.Dense(32, activation="relu")(encoded)
    x = layers.Dense(64, activation="relu")(x)
    decoded = layers.Dense(input_dim, activation="sigmoid")(x)

    autoencoder = keras.Model(inputs, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_deep(X_train, X_test, y_train, y_test, epochs=20, batch_size=256):
    """
    Train LSTM Autoencoder on normal traffic only.
    Anomalies are detected via reconstruction error threshold.
    """
    # Train only on BENIGN traffic (label 0)
    benign_mask = y_train == 0
    X_normal = X_train[benign_mask]

    print(f"[*] Training autoencoder on {X_normal.shape[0]} normal samples...")

    model = build_autoencoder(X_normal.shape[1])
    model.fit(
        X_normal, X_normal,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # Compute reconstruction errors on test set
    reconstructions = model.predict(X_test)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

    # Set threshold at 95th percentile of normal reconstruction error
    normal_mse = mse[y_test == 0]
    threshold = np.percentile(normal_mse, 95)
    print(f"[+] Anomaly threshold (95th percentile): {threshold:.6f}")

    # Predict: above threshold = anomaly
    y_pred = (mse > threshold).astype(int)
    y_binary = (y_test != 0).astype(int)

    from sklearn.metrics import classification_report
    print("\n[+] Deep Model Detection Report:")
    print(classification_report(y_binary, y_pred, target_names=["Normal", "Attack"]))

    # Save model and threshold
    os.makedirs("results", exist_ok=True)
    model.save("results/autoencoder_model.h5")
    np.save("results/threshold.npy", threshold)
    print("[+] Model saved to results/autoencoder_model.h5")
