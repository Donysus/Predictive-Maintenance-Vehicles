import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Data Simulation with Noise Augmentation ---
def generate_sensor_data(n_points=1000, failure_start=700, noise_level=0.5):
    time = np.arange(n_points)
    sensor = np.ones(n_points) * 50 + np.random.normal(0, noise_level, n_points)
    # Add drift after failure_start
    drift = np.linspace(0, 20, n_points - failure_start)
    sensor[failure_start:] += drift + np.random.normal(0, noise_level, n_points - failure_start)
    return time, sensor

time, sensor_data = generate_sensor_data(noise_level=1.0)  # Higher noise for realism

#  Data Splitting (Ensure no failure data in training)
window_size = 50  # Increased to capture longer trends
failure_start = 700
split_time = failure_start - window_size  # 700 - 50 = 650
train_data = sensor_data[:split_time]
test_data = sensor_data[split_time:]

#  Scale Data Using Training Set
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

# Create Windowed Dataset
def create_dataset(data, window_size=50):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_scaled, window_size)
X_test, y_test = create_dataset(test_scaled, window_size)

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], window_size, 1)
X_test = X_test.reshape(X_test.shape[0], window_size, 1)

#  Enhanced LSTM Model
model = Sequential([
    Bidirectional(LSTM(100, return_sequences=True, activation='tanh'), input_shape=(window_size, 1)),
    Dropout(0.2),
    Bidirectional(LSTM(50, activation='tanh')),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

#  Train with Early Stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

#  Save Model & Scaler for Deployment
save_model(model, 'predictive_maintenance_model.keras')
joblib.dump(scaler, 'scaler.save')

#  Anomaly Detection with Consecutive Checks
def detect_anomalies(errors, threshold, consecutive_steps=3):
    anomalies = errors > threshold
    # Flag only if anomalies occur consecutively
    anomaly_streaks = np.zeros_like(anomalies, dtype=bool)
    for i in range(len(anomalies) - consecutive_steps + 1):
        if np.all(anomalies[i:i+consecutive_steps]):
            anomaly_streaks[i:i+consecutive_steps] = True
    return np.where(anomaly_streaks)[0]

# Training errors for threshold
y_train_pred = model.predict(X_train).flatten()
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_train_pred_inv = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
train_errors = np.abs(y_train_inv - y_train_pred_inv)
threshold = np.percentile(train_errors, 99)  # 99th percentile . You can change the percentile as per your need

# Test predictions
y_pred = model.predict(X_test).flatten()
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
test_errors = np.abs(y_test_inv - y_pred_inv)

# Get anomaly indices with consecutive checks
anomaly_indices = detect_anomalies(test_errors, threshold, consecutive_steps=3)
print("Reliable Failure Detections at Test Indices:", anomaly_indices)

#  Visualize Anomalies in Original Time Context
# Map test indices to original time
original_test_time = time[split_time + window_size : split_time + window_size + len(y_test)]
anomaly_times = original_test_time[anomaly_indices]

plt.figure(figsize=(12, 5))
plt.plot(time, sensor_data, label="Sensor Reading")
plt.scatter(anomaly_times, sensor_data[anomaly_indices + split_time + window_size],
            color='red', s=40, label="Anomalies (Failure)")
plt.axvline(x=failure_start, color='black', linestyle='--', label="Failure Onset (True)")
plt.xlabel("Time")
plt.ylabel("Sensor Value")
plt.title("Anomaly Detection in Original Time Context")
plt.legend()
plt.show()
