import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from preprocessor import load_data, preprocess

SEQ_LEN = 30

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        batch_size, seq_len, _ = x.size()
        dec_input = torch.zeros(batch_size, seq_len, self.hidden_dim)
        out, _ = self.decoder(dec_input, (h, c))
        return self.fc(out)

def create_sequences(X, seq_len=SEQ_LEN):
    seqs = []
    for i in range(len(X) - seq_len):
        seqs.append(X[i:i+seq_len])
    return np.array(seqs)

def train_lstm(X, epochs=20, batch_size=64):
    print("Creating sequences...")
    seqs = create_sequences(X)
    tensor = torch.FloatTensor(seqs)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    model = LSTMAutoencoder(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    print("Training LSTM Autoencoder for " + str(epochs) + " epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print("Epoch " + str(epoch+1) + "/" + str(epochs) + " - Loss: " + str(round(total_loss/len(loader), 6)))
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/lstm_autoencoder.pt")
    print("LSTM model saved.")
    return model

def get_reconstruction_errors(model, X):
    model.eval()
    seqs = create_sequences(X)
    tensor = torch.FloatTensor(seqs)
    with torch.no_grad():
        output = model(tensor)
    errors = torch.mean((tensor - output) ** 2, dim=(1, 2)).numpy()
    return errors

def evaluate_lstm(model, X, df):
    print("Calculating reconstruction errors...")
    errors = get_reconstruction_errors(model, X)
    threshold = np.percentile(errors, 95)
    print("Anomaly threshold (95th percentile): " + str(round(float(threshold), 6)))
    preds = (errors > threshold).astype(int)
    pad = np.zeros(SEQ_LEN)
    preds_full = np.concatenate([pad, preds])[:len(df)]
    errors_full = np.concatenate([pad, errors])[:len(df)]
    y_true = df["is_anomaly"].values[:len(preds_full)]
    print("\nClassification Report:")
    print(classification_report(y_true, preds_full, target_names=["Normal","Anomaly"]))
    cm = confusion_matrix(y_true, preds_full)
    os.makedirs("../reports", exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"])
    plt.title("LSTM Autoencoder - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("../reports/lstm_confusion_matrix.png")
    plt.close()
    df2 = df.copy()
    df2["lstm_error"] = errors_full
    df2["lstm_anomaly"] = preds_full
    plt.figure(figsize=(14, 4))
    plt.plot(df2["timestamp"], df2["lstm_error"], label="Reconstruction Error", alpha=0.7)
    plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
    plt.legend()
    plt.title("LSTM - Reconstruction Error over Time")
    plt.tight_layout()
    plt.savefig("../reports/lstm_error_plot.png")
    plt.close()
    print("LSTM plots saved.")
    return preds_full, errors_full, threshold

if __name__ == "__main__":
    df = load_data()
    X, df, cols = preprocess(df, fit=False)
    model = train_lstm(X, epochs=20)
    evaluate_lstm(model, X, df)