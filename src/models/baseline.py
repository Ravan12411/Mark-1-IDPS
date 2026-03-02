import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import os

# Load dataset
df = pd.read_csv("data/raw/cicids2017/cicids2017.csv", low_memory=False)

# Inspect columns to confirm names
print("Columns in dataset:", df.columns.tolist())

# Drop non-predictive columns (ignore if not present)
drop_cols = ["Destination Port", " Destination Port"]  # handle both cases
df = df.drop(columns=drop_cols, errors="ignore")

# Separate features and target (note the leading space in ' Label')
X = df.drop(columns=[" Label"])
y = df[" Label"].astype("category").cat.codes

# Remove classes with fewer than 2 samples (fix stratify error)
class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 2].index
mask = ~y.isin(rare_classes)
X = X[mask]
y = y[mask]

# Clean up infinities and NaNs
X = X.replace([float("inf"), float("-inf")], pd.NA)
X = X.fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with balanced class weights
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

# Random Forest with balanced class weights
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)

# Evaluation function
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall:", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_true, y_pred, average="weighted"))

# Evaluate both models
evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

# Save metrics to results file
os.makedirs("results", exist_ok=True)
with open("results/baseline_metrics.txt", "w") as f:
    f.write("Logistic Regression Results:\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_lr)}\n")
    f.write(f"Precision: {precision_score(y_test, y_pred_lr, average='weighted')}\n")
    f.write(f"Recall: {recall_score(y_test, y_pred_lr, average='weighted')}\n")
    f.write(f"F1 Score: {f1_score(y_test, y_pred_lr, average='weighted')}\n\n")

    f.write("Random Forest Results:\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}\n")
    f.write(f"Precision: {precision_score(y_test, y_pred_rf, average='weighted')}\n")
    f.write(f"Recall: {recall_score(y_test, y_pred_rf, average='weighted')}\n")
    f.write(f"F1 Score: {f1_score(y_test, y_pred_rf, average='weighted')}\n\n")

    f.write("RandomizedSearchCV Best Parameters:\n")
    f.write(str(random_search.best_params_) + "\n")
    f.write(f"Best CV Score: {random_search.best_score_}\n")