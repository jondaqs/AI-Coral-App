# coral_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import os

# --- Coral reef dynamics model ---
def reef_model(y, t, r_C, r_M, r_H, d_C, g, K_H, m_H):
    C, M, H = y
    dCdt = r_C * C * (1 - C - M) - d_C * C
    dMdt = r_M * M * (1 - C - M) - g * H * M
    dHdt = r_H * H * (1 - H / K_H) - m_H * H
    return [dCdt, dMdt, dHdt]

# --- Create and populate database with synthetic data ---
def generate_and_save_to_db(db_file='reef_data.db'):
    if os.path.exists(db_file):
        return  # Skip regeneration if DB already exists

    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS reef_training_data (
            r_C REAL, r_M REAL, r_H REAL, d_C REAL, g REAL, K_H REAL, m_H REAL,
            coral_series TEXT
        )
    ''')

    t = np.linspace(0, 100, 200)
    y0 = [0.3, 0.5, 0.2]
    np.random.seed(42)

    for _ in range(1000):
        params = np.random.uniform(
            [0.2, 0.2, 0.1, 0.05, 0.2, 0.8, 0.01],
            [0.6, 0.8, 0.3, 0.2, 0.9, 1.5, 0.1]
        )
        sol = odeint(reef_model, y0, t, args=tuple(params))
        coral_series = sol[:, 0].tolist()
        coral_str = ','.join(map(str, coral_series))
        c.execute('INSERT INTO reef_training_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                  (*params, coral_str))

    conn.commit()
    conn.close()

# --- Load training data from SQLite ---
def load_data_from_db(db_file='reef_data.db'):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT r_C, r_M, r_H, d_C, g, K_H, m_H, coral_series FROM reef_training_data")
    rows = c.fetchall()
    conn.close()

    X = []
    Y = []
    for row in rows:
        params = list(row[:7])
        coral_series = [float(x) for x in row[7].split(',')]
        X.append(params)
        Y.append(coral_series)

    return np.array(X), np.array(Y)

# --- Train the neural network model to predict full coral time series ---
def train_nn_full_series(X, Y):
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=3000)
    model.fit(X_scaled, Y_scaled)
    return model, scaler_X, scaler_Y

# --- Simulate coral dynamics and predict coral time series ---
def simulate_and_predict(params, y0, t, model, scaler_X, scaler_Y):
    sol = odeint(reef_model, y0, t, args=tuple(params))
    coral_mech = sol[:, 0]

    X_scaled = scaler_X.transform([params])
    pred_scaled = model.predict(X_scaled).reshape(1, -1)
    coral_pred = scaler_Y.inverse_transform(pred_scaled)[0]

    return coral_mech, coral_pred

# --- MAIN APP ---

st.title("🌊 Coral Reef Dynamics: Mechanistic vs Neural Network")

# Generate and load data
with st.spinner("Generating data and loading model..."):
    generate_and_save_to_db()
    X, Y = load_data_from_db()
    model, scaler_X, scaler_Y = train_nn_full_series(X, Y)
    t = np.linspace(0, 100, Y.shape[1])
    y0 = [0.3, 0.5, 0.2]

# Sidebar sliders
st.sidebar.header("🧪 Model Parameters")
r_C = st.sidebar.slider("Coral Growth (r_C)", 0.2, 0.6, 0.4, 0.01)
r_M = st.sidebar.slider("Algae Growth (r_M)", 0.2, 0.8, 0.5, 0.01)
r_H = st.sidebar.slider("Herbivore Growth (r_H)", 0.1, 0.3, 0.2, 0.01)
d_C = st.sidebar.slider("Coral Death (d_C)", 0.05, 0.2, 0.1, 0.01)
g = st.sidebar.slider("Grazing Rate (g)", 0.2, 0.9, 0.6, 0.01)
K_H = st.sidebar.slider("Herbivore Cap (K_H)", 0.8, 1.5, 1.2, 0.05)
m_H = st.sidebar.slider("Herbivore Mortality (m_H)", 0.01, 0.1, 0.05, 0.005)

params = [r_C, r_M, r_H, d_C, g, K_H, m_H]
coral_mech, coral_pred = simulate_and_predict(params, y0, t, model, scaler_X, scaler_Y)

# Plot output
st.subheader("📈 Coral Cover Over Time")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, coral_mech, label='Mechanistic Model', color='deepskyblue')
ax.plot(t, coral_pred, label='NN Predicted Series', linestyle='--', color='purple')
ax.set_xlabel("Time")
ax.set_ylabel("Coral Cover")
ax.legend()
ax.grid(True)
st.pyplot(fig)
