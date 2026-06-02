import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_cloud_metrics(n_samples=5000, anomaly_ratio=0.05, save=True):
    np.random.seed(42)
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i*5) for i in range(n_samples)]
    cpu = np.random.normal(45, 10, n_samples).clip(0, 100)
    memory = np.random.normal(60, 8, n_samples).clip(0, 100)
    network = np.random.normal(200, 40, n_samples).clip(0, 1000)
    disk_io = np.random.normal(100, 20, n_samples).clip(0, 500)
    response_time = np.random.normal(150, 30, n_samples).clip(0, 2000)
    error_rate = np.random.normal(1, 0.5, n_samples).clip(0, 100)
    labels = np.zeros(n_samples)
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    for idx in anomaly_indices:
        spike_type = np.random.randint(0, 4)
        if spike_type == 0:
            cpu[idx] = np.random.uniform(90, 100)
            memory[idx] = np.random.uniform(88, 100)
        elif spike_type == 1:
            network[idx] = np.random.uniform(800, 1000)
            response_time[idx] = np.random.uniform(1500, 2000)
        elif spike_type == 2:
            disk_io[idx] = np.random.uniform(400, 500)
            error_rate[idx] = np.random.uniform(20, 100)
        else:
            cpu[idx] = np.random.uniform(85, 100)
            memory[idx] = np.random.uniform(85, 100)
            network[idx] = np.random.uniform(700, 1000)
            error_rate[idx] = np.random.uniform(15, 100)
        labels[idx] = 1
    df = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_usage": cpu.round(2),
        "memory_usage": memory.round(2),
        "network_traffic": network.round(2),
        "disk_io": disk_io.round(2),
        "response_time": response_time.round(2),
        "error_rate": error_rate.round(2),
        "is_anomaly": labels.astype(int)
    })
    os.makedirs("data", exist_ok=True)
    if save:
        df.to_csv("data/cloud_metrics.csv", index=False)
        print(f"Dataset saved: {len(df)} rows, {int(labels.sum())} anomalies")
    return df

if __name__ == "__main__":
    df = generate_cloud_metrics()
    print(df.head())
    print(f"Anomaly rate: {df.is_anomaly.mean()*100:.1f}%")
