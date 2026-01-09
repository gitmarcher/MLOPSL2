import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ PATHS ------------------
DATA_PATH = "dataset/winequality-red.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ LOAD DATA ------------------
df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# ------------------ TRAIN-TEST SPLIT (60/40) ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# ------------------ MODEL ------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train)

# ------------------ EVALUATION ------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ------------------ OUTPUT ------------------
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

joblib.dump(model, MODEL_PATH)

results = {
    "experiment_id": "EXP-07",
    "model": "Random Forest",
    "hyperparameters": "n_estimators=100, max_depth=12",
    "preprocessing": "None",
    "feature_selection": "All",
    "split": "60/40",
    "mse": mse,
    "r2_score": r2
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)
