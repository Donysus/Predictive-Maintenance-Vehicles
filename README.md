# AI‑Based Predictive Maintenance for Vehicles:

This project demonstrates an AI‑based predictive maintenance system for vehicles using an enhanced LSTM model with Bidirectional layers. The project simulates vehicle sensor data with noise and drift, trains a model to predict future sensor readings, and detects anomalies to flag potential failures.

## Features:

- **Data Simulation:** Realistic sensor data generation with noise and drift.
- **Enhanced LSTM Model:** Bidirectional LSTM layers with dropout to capture long‑term dependencies.
- **Anomaly Detection:** Consecutive anomaly checking with a threshold based on the 99th percentile of training errors.
- **Deployment Ready:** Saves the trained model and scaler for production use.

## Installation:

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage:

Run the main script:

```bash
python predictive_maintenance.py
```

## Files:

1. predictive_maintenance.py - Main project code.
2. requirements.txt - List of dependencies.
3. .gitignore - Git ignore file.

## Author

Raghav Jha

raghavmrparadise@gmail.com
