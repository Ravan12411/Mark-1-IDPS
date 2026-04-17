"""
Mark-1 IDPS — AI/ML Intrusion Detection & Prevention System
CLI entry point for training, evaluation, and live detection.
"""

import argparse
import os
import sys

def run_preprocess():
    print("[*] Loading and preprocessing CICIDS2017 dataset...")
    from src.data.preprocess import preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess()
    print("[+] Preprocessing complete.")
    return X_train, X_test, y_train, y_test, scaler

def run_train(args):
    X_train, X_test, y_train, y_test, scaler = run_preprocess()
    if args.model == "baseline":
        print("[*] Training baseline models (Logistic Regression + Random Forest)...")
        from src.models.baseline import train_baseline
        train_baseline(X_train, X_test, y_train, y_test)
    elif args.model == "deep":
        print("[*] Training deep learning model (LSTM Autoencoder)...")
        from src.models.train import train_deep
        train_deep(X_train, X_test, y_train, y_test)
    else:
        print("[-] Unknown model type. Use --model baseline or --model deep")
        sys.exit(1)

def run_detect(args):
    if not args.input:
        print("[-] Please provide an input CSV file with --input <path>")
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"[-] File not found: {args.input}")
        sys.exit(1)
    print(f"[*] Running anomaly detection on: {args.input}")
    from src.detection.detector import detect
    detect(args.input)

def run_evaluate(args):
    X_train, X_test, y_train, y_test, scaler = run_preprocess()
    print("[*] Evaluating models...")
    from src.models.evaluate import evaluate_all
    evaluate_all(X_test, y_test)

def main():
    parser = argparse.ArgumentParser(
        description="Mark-1 IDPS — AI/ML Anomaly-Based Intrusion Detection System"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train detection models")
    train_parser.add_argument(
        "--model",
        choices=["baseline", "deep"],
        default="baseline",
        help="Model type to train (default: baseline)"
    )

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Run detection on network traffic CSV")
    detect_parser.add_argument("--input", type=str, help="Path to input CSV file")

    # Evaluate command
    subparsers.add_parser("evaluate", help="Evaluate trained models and print metrics")

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "detect":
        run_detect(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
