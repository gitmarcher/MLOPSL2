import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ PATHS ------------------
DATA_PATH = "dataset/winequality-red.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ LOAD DATA ------------------
df = pd.read_csv(DATA_PATH, sep=";")

y = df["quality"]

# ------------------ CORRELATION-BASED FEATURE SELECTION ------------------
corr_features = (
    df.corr()["quality"]
    .abs()
    .sort_values(ascending=False)
    .index[1:7]
)

X = df[corr_features]

# ------------------ TRAIN-TEST SPLIT (80/20) ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ STANDARDIZATION ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ MODEL ------------------
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ------------------ EVALUATION ------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ------------------ OUTPUT ------------------
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

joblib.dump(
    {"model": model, "scaler": scaler, "features": list(corr_features)},
    MODEL_PATH
)

results = {
    "experiment_id": "EXP-04",
    "model": "Ridge Regression",
    "hyperparameters": "alpha=1.0",
    "preprocessing": "Standardization",
    "feature_selection": "Correlation-based (top 6)",
    "split": "80/20",
    "mse": mse,
    "r2_score": r2
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)
