import os
import json
import joblib
import pandas as pd
import numpy as np

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

# ------------------ INITIAL MODEL FOR FEATURE IMPORTANCE ------------------
base_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
base_model.fit(X, y)

# ------------------ IMPORTANCE-BASED FEATURE SELECTION ------------------
importances = base_model.feature_importances_
top_indices = np.argsort(importances)[-6:]
important_features = X.columns[top_indices]

X_imp = X[important_features]

# ------------------ TRAIN-TEST SPLIT (80/20) ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42
)

# ------------------ FINAL MODEL ------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
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

joblib.dump(
    {
        "model": model,
        "selected_features": list(important_features)
    },
    MODEL_PATH
)

results = {
    "experiment_id": "EXP-06",
    "model": "Random Forest",
    "hyperparameters": "n_estimators=100, max_depth=15",
    "preprocessing": "None",
    "feature_selection": "Importance-based (top 6)",
    "split": "80/20",
    "mse": mse,
    "r2_score": r2
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)
