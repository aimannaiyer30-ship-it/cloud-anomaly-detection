import numpy as np
import pandas as pd
import pickle
import torch
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from preprocessor import load_data, preprocess
from isolation_forest_model import train_isolation_forest, evaluate as if_evaluate
from lstm_model import LSTMAutoencoder, train_lstm, evaluate_lstm, get_reconstruction_errors, SEQ_LEN

def run_full_evaluation():
    print("="*50)
    print("FULL SYSTEM EVALUATION")
    print("="*50)
    df = load_data()
    X, df, cols = preprocess(df, fit=True)

    print("\n--- Isolation Forest ---")
    if_model = train_isolation_forest(X)
    if_preds, if_scores = if_evaluate(if_model, X, df)

    print("\n--- LSTM Autoencoder ---")
    lstm_model = train_lstm(X, epochs=20)
    lstm_preds, lstm_errors, threshold = evaluate_lstm(lstm_model, X, df)

    print("\n--- Saving Results ---")
    df["if_anomaly"] = if_preds
    df["if_score"] = if_scores
    pad = np.zeros(SEQ_LEN)
    df["lstm_anomaly"] = np.concatenate([pad, lstm_preds])[:len(df)]
    df["lstm_error"] = np.concatenate([pad, lstm_errors])[:len(df)]
    df.to_csv("../reports/results.csv", index=False)

    y_true = df["is_anomaly"].values
    print("\n=== FINAL SUMMARY ===")
    print(f"Isolation Forest F1: {f1_score(y_true, if_preds, zero_division=0):.4f}")
    print(f"Isolation Forest Precision: {precision_score(y_true, if_preds, zero_division=0):.4f}")
    print(f"Isolation Forest Recall: {recall_score(y_true, if_preds, zero_division=0):.4f}")
    lstm_full = df["lstm_anomaly"].astype(int).values
    print(f"LSTM F1: {f1_score(y_true, lstm_full, zero_division=0):.4f}")
    print(f"LSTM Precision: {precision_score(y_true, lstm_full, zero_division=0):.4f}")
    print(f"LSTM Recall: {recall_score(y_true, lstm_full, zero_division=0):.4f}")
    print("\nAll results saved to reports/")

if __name__ == "__main__":
    run_full_evaluation()