import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessor import load_data, preprocess, FEATURES
from isolation_forest_model import train_isolation_forest
from lstm_model import LSTMAutoencoder, train_lstm, get_reconstruction_errors, create_sequences, SEQ_LEN
from alert_system import generate_alerts
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Cloud Anomaly Detection", layout="wide", page_icon="🔍")
st.title("🔍 AI-Based Anomaly Detection for Cloud Resources")
st.markdown("Monitors cloud metrics using **Isolation Forest** and **LSTM Autoencoder** to detect anomalies.")

@st.cache_data
def load_and_prep():
    df = load_data(os.path.join(os.path.dirname(__file__), "..", "data", "cloud_metrics.csv"))
    X, df, cols = preprocess(df, fit=True)
    return df, X, cols

@st.cache_resource
def get_if_model(X):
    model = train_isolation_forest(X)
    return model

@st.cache_resource
def get_lstm_model(X):
    model = train_lstm(X, epochs=20)
    return model

with st.spinner("Loading data and training models..."):
    df, X, cols = load_and_prep()
    if_model = get_if_model(X)
    lstm_model = get_lstm_model(X)

if_preds = np.where(if_model.predict(X) == -1, 1, 0)
if_scores = -if_model.score_samples(X)
df["if_anomaly"] = if_preds
df["if_score"] = if_scores

errors = get_reconstruction_errors(lstm_model, X)
threshold = np.percentile(errors, 95)
lstm_preds = (errors > threshold).astype(int)
pad = np.zeros(SEQ_LEN)
df["lstm_anomaly"] = np.concatenate([pad, lstm_preds])[:len(df)]
df["lstm_error"] = np.concatenate([pad, errors])[:len(df)]

st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox("Select Model", ["Isolation Forest", "LSTM Autoencoder", "Both"])
metric_choice = st.sidebar.selectbox("Select Metric", FEATURES)
show_only_anomalies = st.sidebar.checkbox("Show only anomalies in table")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(df))
col2.metric("IF Anomalies", int(df["if_anomaly"].sum()))
col3.metric("LSTM Anomalies", int(df["lstm_anomaly"].sum()))
col4.metric("True Anomalies", int(df["is_anomaly"].sum()))

st.subheader(f"📈 {metric_choice} over Time")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df[metric_choice], mode="lines", name=metric_choice, line=dict(color="#4C72B0", width=1)))

if model_choice in ["Isolation Forest", "Both"]:
    anom = df[df["if_anomaly"]==1]
    fig.add_trace(go.Scatter(x=anom["timestamp"], y=anom[metric_choice], mode="markers", name="IF Anomaly", marker=dict(color="red", size=6, symbol="x")))

if model_choice in ["LSTM Autoencoder", "Both"]:
    anom2 = df[df["lstm_anomaly"]==1]
    fig.add_trace(go.Scatter(x=anom2["timestamp"], y=anom2[metric_choice], mode="markers", name="LSTM Anomaly", marker=dict(color="orange", size=6, symbol="circle")))

fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
st.plotly_chart(fig, use_container_width=True)

if model_choice in ["LSTM Autoencoder", "Both"]:
    st.subheader("📉 LSTM Reconstruction Error")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["lstm_error"], mode="lines", name="Reconstruction Error", line=dict(color="#DD8452")))
    fig2.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig2.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("📊 Metric Distribution")
fig3 = px.box(df, y=FEATURES, points=False)
fig3.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0))
st.plotly_chart(fig3, use_container_width=True)

st.subheader("🚨 Recent Alerts")
alert_col = "if_anomaly" if model_choice == "Isolation Forest" else "lstm_anomaly"
alerts_df = df[df[alert_col]==1][["timestamp","cpu_usage","memory_usage","network_traffic","disk_io","response_time","error_rate","is_anomaly"]].tail(20)
if show_only_anomalies:
    st.dataframe(alerts_df, use_container_width=True)
else:
    st.dataframe(df[["timestamp","cpu_usage","memory_usage","network_traffic","disk_io","response_time","error_rate","is_anomaly"]].tail(50), use_container_width=True)

st.subheader("📋 Raw Data Sample")
st.dataframe(df[FEATURES + ["is_anomaly","if_anomaly","lstm_anomaly"]].head(100), use_container_width=True)
