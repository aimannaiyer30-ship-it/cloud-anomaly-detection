import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

FEATURES = ["cpu_usage", "memory_usage", "network_traffic", "disk_io", "response_time", "error_rate"]

def load_data(path="data/cloud_metrics.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows")
    return df

def preprocess(df, fit=True, scaler_path="models/scaler.pkl"):
    df = df.copy()
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    for f in FEATURES:
        df[f"rolling_mean_{f}"] = df[f].rolling(window=10, min_periods=1).mean()
        df[f"rolling_std_{f}"] = df[f].rolling(window=10, min_periods=1).std().fillna(0)
    feature_cols = FEATURES + [f"rolling_mean_{f}" for f in FEATURES] + [f"rolling_std_{f}" for f in FEATURES]
    X = df[feature_cols].values
    os.makedirs("models", exist_ok=True)
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print("Scaler saved.")
    else:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
    return X_scaled, df, feature_cols

if __name__ == "__main__":
    df = load_data()
    X, df_out, cols = preprocess(df)
    print(f"Feature matrix shape: {X.shape}")
