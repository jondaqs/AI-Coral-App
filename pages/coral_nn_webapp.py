# coral_nn_webapp.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# Coral reef dynamics model
def reef_model(y, t, r_C, r_M, r_H, d_C, g, K_H, m_H):
    C, M, H = y
    dCdt = r_C * C * (1 - C - M) - d_C * C
    dMdt = r_M * M * (1 - C - M) - g * H * M
    dHdt = r_H * H * (1 - H / K_H) - m_H * H
    return [dCdt, dMdt, dHdt]

# Train the neural network model
def train_nn_model(X, y):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1, 1)).ravel()
    model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', max_iter=2000)
    model.fit(X_scaled, y_scaled)
    return model, scaler_X, scaler_y

# Predict final coral cover
def predict_with_nn(model, scaler_X, scaler_y, params):
    X = scaler_X.transform([params])
    y_pred_scaled = model.predict(X)
    return scaler_y.inverse_transform([[y_pred_scaled[0]]])[0][0]

# Generate training data and train model
@st.cache_data
def prepare_model():
    t = np.linspace(0, 100, 200)
    y0 = [0.3, 0.5, 0.2]
    np.random.seed(0)
    samples = 1000
    params_all = np.random.uniform(
        [0.2, 0.2, 0.1, 0.05, 0.2, 0.8, 0.01],
        [0.6, 0.8, 0.3, 0.2, 0.9, 1.5, 0.1],
        (samples, 7)
    )
    coral_final = []
    for params in params_all:
        sol = odeint(reef_model, y0, t, args=tuple(params))
        coral_final.append(sol[-1, 0])
    model, scaler_X, scaler_y = train_nn_model(params_all, coral_final)
    return model, scaler_X, scaler_y, t, y0

# Load and train model
st.title("🌊 Coral Reef Dynamics + Neural Network Prediction")
with st.spinner("Training neural network..."):
    nn_model, scaler_X, scaler_y, t_global, y0_global = prepare_model()

# Sidebar sliders for parameters
st.sidebar.header("🧪 Adjust Model Parameters")
r_C = st.sidebar.slider("Coral Growth (r_C)", 0.2, 0.6, 0.4, 0.01)
r_M = st.sidebar.slider("Algae Growth (r_M)", 0.2, 0.8, 0.5, 0.01)
r_H = st.sidebar.slider("Herbivore Growth (r_H)", 0.1, 0.3, 0.2, 0.01)
d_C = st.sidebar.slider("Coral Death (d_C)", 0.05, 0.2, 0.1, 0.01)
g = st.sidebar.slider("Grazing Rate (g)", 0.2, 0.9, 0.6, 0.01)
K_H = st.sidebar.slider("Herbivore Cap (K_H)", 0.8, 1.5, 1.2, 0.05)
m_H = st.sidebar.slider("Herbivore Mortality (m_H)", 0.01, 0.1, 0.05, 0.005)

# Simulate and predict
params = [r_C, r_M, r_H, d_C, g, K_H, m_H]
sol = odeint(reef_model, y0_global, t_global, args=tuple(params))
final_pred = predict_with_nn(nn_model, scaler_X, scaler_y, params)

# Plot results
st.subheader("📈 Reef Simulation vs NN Final Prediction")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_global, sol[:, 0], label='Coral (C)', color='deepskyblue')
ax.plot(t_global, sol[:, 1], label='Algae (M)', color='forestgreen')
ax.plot(t_global, sol[:, 2], label='Herbivores (H)', color='orange')
ax.axhline(final_pred, color='blue', linestyle='--',
           label=f'NN Predicted Final Coral: {final_pred:.2f}')
ax.set_xlabel('Time')
ax.set_ylabel('Proportion / Biomass')
ax.set_title("Coral Reef Dynamics")
ax.legend()
ax.grid(True)
st.pyplot(fig)
