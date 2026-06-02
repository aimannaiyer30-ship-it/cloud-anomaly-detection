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

st.set_page_config(
    page_title="Cloud Anomaly Detection | Aiman Nadira Naiyer",
    layout="wide",
    page_icon="chart_with_upwards_trend"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1.5rem; }
    .student-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .student-header h2 { margin: 0; font-size: 22px; color: #e0e0e0; }
    .student-header p { margin: 4px 0 0 0; font-size: 14px; color: #a0a0c0; }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #4361ee;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .metric-card h3 { margin: 0; font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card p { margin: 6px 0 0 0; font-size: 28px; font-weight: 600; color: #1a1a2e; }
    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #1a1a2e;
        margin: 20px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #e0e0e0;
    }
    .alert-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 16px;
        border-radius: 4px;
        font-size: 13px;
        margin-bottom: 8px;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #999;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="student-header">
        <h2>AI-Based Anomaly Detection System for Cloud Resource Monitoring</h2>
        <p>Aiman Nadira Naiyer &nbsp;|&nbsp; MCA Cloud Computing &nbsp;|&nbsp; Chandigarh University &nbsp;|&nbsp; Final Year Project 2024-25</p>
    </div>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cloud_metrics.csv")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
IF_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "isolation_forest.pkl")
LSTM_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lstm_autoencoder.pt")

@st.cache_data
def load_and_prep():
    df = load_data(DATA_PATH)
    X, df, cols = preprocess(df, fit=True, scaler_path=SCALER_PATH)
    return df, X, cols

@st.cache_resource
def get_if_model(X):
    return train_isolation_forest(X)

@st.cache_resource
def get_lstm_model(X):
    return train_lstm(X, epochs=20)

with st.spinner("Loading models and analyzing cloud metrics..."):
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

st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/6/sixth/Chandigarh_University_seal.png/180px-Chandigarh_University_seal.png", width=80) if False else None
st.sidebar.markdown("### Control Panel")
st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("Detection Model", ["Isolation Forest", "LSTM Autoencoder", "Both Models"])
metric_choice = st.sidebar.selectbox("Metric to Display", FEATURES)
time_range = st.sidebar.slider("Show last N records", min_value=500, max_value=5000, value=2000, step=500)
show_anomalies_only = st.sidebar.checkbox("Highlight anomalies only")
st.sidebar.markdown("---")
st.sidebar.markdown("**About this project**")
st.sidebar.markdown("This system uses Isolation Forest and LSTM Autoencoder to detect unusual patterns in cloud resource usage metrics.")

df_view = df.tail(time_range).copy()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown('<div class="metric-card"><h3>Total Records</h3><p>' + str(len(df)) + '</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>Time Range</h3><p>' + str(len(df_view)) + '</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>IF Anomalies</h3><p>' + str(int(df_view["if_anomaly"].sum())) + '</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><h3>LSTM Anomalies</h3><p>' + str(int(df_view["lstm_anomaly"].sum())) + '</p></div>', unsafe_allow_html=True)
with col5:
    st.markdown('<div class="metric-card"><h3>Actual Anomalies</h3><p>' + str(int(df_view["is_anomaly"].sum())) + '</p></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Metric Trend with Anomaly Markers</div>', unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_view["timestamp"], y=df_view[metric_choice],
    mode="lines", name=metric_choice.replace("_", " ").title(),
    line=dict(color="#4361ee", width=1.2), opacity=0.8
))

if model_choice in ["Isolation Forest", "Both Models"]:
    anom_if = df_view[df_view["if_anomaly"] == 1]
    fig.add_trace(go.Scatter(
        x=anom_if["timestamp"], y=anom_if[metric_choice],
        mode="markers", name="IF Anomaly",
        marker=dict(color="#e63946", size=7, symbol="x", line=dict(width=1.5))
    ))

if model_choice in ["LSTM Autoencoder", "Both Models"]:
    anom_lstm = df_view[df_view["lstm_anomaly"] == 1]
    fig.add_trace(go.Scatter(
        x=anom_lstm["timestamp"], y=anom_lstm[metric_choice],
        mode="markers", name="LSTM Anomaly",
        marker=dict(color="#f4a261", size=7, symbol="circle-open", line=dict(width=1.5))
    ))

fig.update_layout(
    height=380,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0")
)
st.plotly_chart(fig, use_container_width=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-title">LSTM Reconstruction Error</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_view["timestamp"], y=df_view["lstm_error"],
        mode="lines", name="Reconstruction Error",
        line=dict(color="#f4a261", width=1), fill="tozeroy", fillcolor="rgba(244,162,97,0.1)"
    ))
    fig2.add_hline(y=threshold, line_dash="dash", line_color="#e63946", annotation_text="Anomaly Threshold")
    fig2.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0")
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown('<div class="section-title">Anomaly Score Distribution</div>', unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=df_view["if_score"], nbinsx=50,
        marker_color="#4361ee", opacity=0.7, name="IF Score"
    ))
    fig3.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        xaxis_title="Anomaly Score", yaxis_title="Count"
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('<div class="section-title">All Metrics Overview</div>', unsafe_allow_html=True)
fig4 = px.box(df_view[FEATURES], points=False, color_discrete_sequence=["#4361ee"])
fig4.update_layout(
    height=300, margin=dict(l=0, r=0, t=10, b=0),
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
)
st.plotly_chart(fig4, use_container_width=True)

st.markdown('<div class="section-title">Recent Anomaly Alerts</div>', unsafe_allow_html=True)
alert_col = "if_anomaly" if model_choice == "Isolation Forest" else "lstm_anomaly"
alerts = df_view[df_view[alert_col] == 1][["timestamp","cpu_usage","memory_usage","network_traffic","disk_io","response_time","error_rate"]].tail(10)
if len(alerts) > 0:
    for _, row in alerts.iterrows():
        dominant = row[["cpu_usage","memory_usage","network_traffic","disk_io","response_time","error_rate"]].idxmax()
        st.markdown('<div class="alert-box">Anomaly detected at <b>' + str(row["timestamp"]) + '</b> — High <b>' + str(dominant.replace("_"," ").title()) + '</b>: ' + str(round(row[dominant],2)) + '</div>', unsafe_allow_html=True)
else:
    st.info("No anomalies found in selected range.")

st.markdown('<div class="section-title">Data Table</div>', unsafe_allow_html=True)
display_df = alerts if show_anomalies_only else df_view[["timestamp","cpu_usage","memory_usage","network_traffic","disk_io","response_time","error_rate","is_anomaly","if_anomaly","lstm_anomaly"]].tail(50)
st.dataframe(display_df, use_container_width=True)

st.markdown('<div class="footer">Aiman Nadira Naiyer | MCA Cloud Computing | Chandigarh University | 2024-25<br>AI-Based Anomaly Detection System for Cloud Resource Monitoring</div>', unsafe_allow_html=True)