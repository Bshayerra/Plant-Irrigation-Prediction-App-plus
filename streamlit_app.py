
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained models and LabelEncoder
# Make sure these .pkl files are in the same directory as this script
rf_regressor = joblib.load('rf_regressor.pkl')
rf_regressor_pumptime = joblib.load('rf_regressor_pumptime.pkl')
rf_classifier = joblib.load('logistic_regression_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title('Plant Health Prediction App')
st.subheader('Enter Plant Health Parameters')

soil_moisture_input = st.number_input('Soil_Moisture', min_value=0.0, max_value=100.0, value=25.0)
soil_temperature_input = st.number_input('Soil_Temperature', min_value=0.0, max_value=50.0, value=20.0)
ambient_temperature_input = st.number_input('Ambient_Temperature', min_value=0.0, max_value=50.0, value=22.0)

input_data = pd.DataFrame([[soil_moisture_input, soil_temperature_input, ambient_temperature_input]],
                            columns=['Soil_Moisture', 'Soil_Temperature', 'Ambient_Temperature'])

if st.button('Predict'):
    predicted_wateramount = rf_regressor.predict(input_data)[0]
    predicted_pumptime = rf_regressor_pumptime.predict(input_data)[0]
    predicted_pump_on_encoded = rf_classifier.predict(input_data)[0]
    predicted_pump_on_label = label_encoder.inverse_transform([predicted_pump_on_encoded])[0]

    st.subheader('Prediction Results:')
    st.write(f"Predicted Water Amount: {predicted_wateramount:.2f}")
    st.write(f"Predicted Pump Time: {predicted_pumptime:.2f}")
    st.write(f"Predicted Pump On/Off Status: {predicted_pump_on_label}")
