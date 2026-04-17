# Mark-1 IDPS

**AI/ML Anomaly-Based Intrusion Detection & Prevention System**

Mark-1 uses machine learning to detect network intrusions in real time. It trains on the [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset and classifies network traffic as benign or one of 12 attack types including DoS, DDoS, PortScan, Brute Force, and Web Attacks.

---

## Results

| Model | Accuracy | F1 Score |
| Logistic Regression | 94.0% | 93.8% |
| Random Forest | **99.8%** | **99.8%** |

---

## Architecture

```
Mark-1-IDPS/
├── main.py                  # CLI entry point
├── src/
│   ├── data/
│   │   ├── load_dataset.py  # Dataset inspection
│   │   └── preprocess.py    # Cleaning, encoding, scaling
│   ├── models/
│   │   ├── baseline.py      # Logistic Regression + Random Forest
│   │   ├── train.py         # LSTM Autoencoder (deep model)
│   │   └── evaluate.py      # Metrics and evaluation
│   └── detection/
│       └── detector.py      # Live detection on new traffic CSV
└── results/                 # Saved models and metrics
```

---

## Setup

```bash
git clone https://github.com/Ravan12411/Mark-1-IDPS.git
cd Mark-1-IDPS
pip install -r requirements.txt
```

Download CICIDS2017 dataset and place it at:
```
data/raw/cicids2017/cicids2017.csv
```

---

## Usage

**Train baseline models (Logistic Regression + Random Forest):**
```bash
python main.py train --model baseline
```

**Train deep model (LSTM Autoencoder):**
```bash
python main.py train --model deep
```

**Run detection on new traffic:**
```bash
python main.py detect --input path/to/traffic.csv
```

**Evaluate trained models:**
```bash
python main.py evaluate
```

---

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity  
2.8M network flow records across 15 attack categories.

---

## License

Unlicense — public domain.
