import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessor import load_data, preprocess

def train_isolation_forest(X, contamination=0.05):
    print("Training Isolation Forest...")
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X)
    os.makedirs("../models", exist_ok=True)
    with open("../models/isolation_forest.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved.")
    return model

def evaluate(model, X, df):
    preds = model.predict(X)
    preds_binary = np.where(preds == -1, 1, 0)
    scores = -model.score_samples(X)
    y_true = df["is_anomaly"].values
    print("\nClassification Report:")
    print(classification_report(y_true, preds_binary, target_names=["Normal", "Anomaly"]))
    cm = confusion_matrix(y_true, preds_binary)
    os.makedirs("../reports", exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"])
    plt.title("Isolation Forest - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("../reports/if_confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved.")
    df["if_score"] = scores
    df["if_anomaly"] = preds_binary
    plt.figure(figsize=(14, 4))
    plt.plot(df["timestamp"], df["cpu_usage"], label="CPU", alpha=0.6)
    anomalies = df[df["if_anomaly"] == 1]
    plt.scatter(anomalies["timestamp"], anomalies["cpu_usage"], color="red", s=20, label="Anomaly", zorder=5)
    plt.legend()
    plt.title("Isolation Forest - Anomaly Detection on CPU Usage")
    plt.tight_layout()
    plt.savefig("../reports/if_anomaly_plot.png")
    plt.close()
    print("Anomaly plot saved.")
    return preds_binary, scores

if __name__ == "__main__":
    df = load_data()
    X, df, cols = preprocess(df, fit=True)
    model = train_isolation_forest(X)
    evaluate(model, X, df)