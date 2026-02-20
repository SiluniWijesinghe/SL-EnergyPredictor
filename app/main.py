import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="LankaGrid-Forecaster", layout="wide")

# Load the saved model and scaler
# Note: Ensure these paths match your folder structure
MODEL_PATH = 'models/trained_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# UI Header
st.title("⚡ LankaGrid-Forecaster")
st.markdown("### National Electricity Load Demand Prediction (Sri Lanka)")
st.write("Adjust the parameters below to predict the 15-minute interval grid load.")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.header("🌦️ Weather Conditions")
    temp = st.slider("Temperature (°C)", 20.0, 40.0, 30.0)
    hum = st.slider("Humidity (%)", 30.0, 100.0, 75.0)
    wind = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 2.5)
    rain = st.number_input("Rainfall (mm)", 0.0, 100.0, 0.0)
    solar = st.number_input("Solar Irradiance (W/m²)", 0.0, 1200.0, 500.0)

with col2:
    st.header("🕒 Temporal & Economic")
    hour = st.selectbox("Hour of Day", list(range(24)))
    day = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))
    month = st.selectbox("Month", list(range(1, 13)))
    price = st.number_input("Electricity Price (LKR/kWh)", 5.0, 100.0, 25.0)
    is_public_event = st.checkbox("Public Event/Holiday?")
    season = st.radio("Season", ["Summer", "Fall", "Winter"])

# Prediction Logic
if st.button("🔮 Predict Load Demand"):
    try:
        model, scaler = load_assets()

        # 1. Prepare input vector (Must match the exact order of your training features)
        # Sequence: Temp, Hum, Wind, Rain, Solar, GDP(mean), PerCapita(mean), Price, Day, Hour, Month, Event, Fall, Summer, Winter, Weekend
        gdp_mean = 1000.0  # Constant or mean from training
        energy_mean = 500.0
        is_weekend = 1 if day >= 5 else 0
        
        s_fall = 1 if season == "Fall" else 0
        s_summer = 1 if season == "Summer" else 0
        s_winter = 1 if season == "Winter" else 0
        is_event = 1 if is_public_event else 0

        input_data = np.array([[temp, hum, wind, rain, solar, gdp_mean, energy_mean, 
                                price, day, hour, month, is_event, s_fall, s_summer, s_winter, is_weekend]])
        
        # 2. Scale inputs
        input_scaled = scaler.transform(input_data)

        # 3. Predict
        prediction = model.predict(input_scaled)[0]

        # 4. Display result
        st.success(f"### Predicted Load: {prediction:.2f} kW")
        
    except Exception as e:
        st.error(f"Error: {e}. Make sure you ran your Notebook to save the model files first!")