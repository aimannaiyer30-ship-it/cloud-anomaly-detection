import pandas as pd
import numpy as np
import os
from datetime import datetime

LOG_FILE = "../reports/alerts.log"

def log_alert(timestamp, metric, value, anomaly_type, score):
    os.makedirs("../reports", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        msg = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] ALERT"
        msg += " | Time: " + str(timestamp)
        msg += " | Metric: " + str(metric)
        msg += " | Value: " + str(round(float(value), 2))
        msg += " | Type: " + str(anomaly_type)
        msg += " | Score: " + str(round(float(score), 4))
        f.write(msg + "\n")
        print(msg)

def generate_alerts(df, score_col, threshold, anomaly_col, model_name="Model"):
    alerts = df[df[anomaly_col] == 1].copy()
    print(str(model_name) + " - " + str(len(alerts)) + " anomalies detected")
    for _, row in alerts.iterrows():
        dominant = row[["cpu_usage","memory_usage","network_traffic","disk_io","response_time","error_rate"]].idxmax()
        log_alert(row["timestamp"], dominant, row[dominant], model_name, row[score_col])
    return alerts

if __name__ == "__main__":
    print("Alert system ready.")