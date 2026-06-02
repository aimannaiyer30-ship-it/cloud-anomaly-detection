# AI-Based Anomaly Detection System for Cloud Resource Monitoring

## Setup
1. Create virtual environment: python -m venv venv
2. Activate: venv\Scripts\activate
3. Install: pip install -r requirements.txt

## Run Steps

### Step 1: Generate Data
cd src
python data_generator.py

### Step 2: Train Models and Evaluate
python evaluator.py

### Step 3: Launch Dashboard
cd ..
streamlit run dashboard/app.py

## Models
- Isolation Forest: Unsupervised anomaly detection
- LSTM Autoencoder: Time-series reconstruction error based detection

## Metrics Monitored
CPU Usage, Memory Usage, Network Traffic, Disk IO, Response Time, Error Rate