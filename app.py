import streamlit as st
import requests
import pandas as pd

# ----------------------------
# Config
# ----------------------------
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.title("🚲 Bike Rental Prediction")

st.write("Enter features and get predicted number of rentals")

# ----------------------------
# Inputs
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    season = st.selectbox("Season", [1, 2, 3, 4])
    year = st.selectbox("Year (0=2011, 1=2012)", [0, 1])
    month = st.slider("Month", 1, 12, 1)
    hour = st.slider("Hour", 0, 23, 12)

with col2:
    holiday = st.selectbox("Holiday", [0, 1])
    working_day = st.selectbox("Working Day", [0, 1])
    week_day = st.slider("Week Day (0=Sun)", 0, 6, 1)
    weather_situation = st.selectbox("Weather", [1, 2, 3, 4])

st.subheader("Weather Conditions")

temp_norm = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5)
feels_like_temp_norm = st.slider("Feels Like Temp", 0.0, 1.0, 0.5)
humidity_norm = st.slider("Humidity", 0.0, 1.0, 0.5)
wind_speed = st.slider("Wind Speed", 0.0, 1.0, 0.2)

st.subheader("User Activity")

casual = st.number_input("Casual Users", min_value=0, value=10)
registered = st.number_input("Registered Users", min_value=0, value=50)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict 🚀"):

    data = [{
        "season": season,
        "year": year,
        "month": month,
        "hour": hour,
        "holiday": holiday,
        "week_day": week_day,
        "working_day": working_day,
        "weather_situation": weather_situation,
        "temp_norm": temp_norm,
        "feels_like_temp_norm": feels_like_temp_norm,
        "humidity_norm": humidity_norm,
        "wind_speed": wind_speed,
        "casual": casual,
        "registered": registered
    }]

    try:
        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            prediction = response.json()["predictions"][0]
            st.success(f"🎯 Predicted Rentals: {prediction:.2f}")
        else:
            st.error(f"Error: {response.text}")

    except Exception as e:
        st.error(f"API not reachable: {e}")