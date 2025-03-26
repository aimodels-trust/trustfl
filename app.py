import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model("/mnt/data/federated_credit_card_model.keras")

# Load additional resources
with open("/mnt/data/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("/mnt/data/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

# Function to preprocess user input
def preprocess_input(input_data, scaler):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    return input_scaled

# Streamlit UI
st.title("Credit Card Default Prediction")

# Single Prediction Form
st.header("Single Prediction")
input_values = {}
for feature in feature_names:
    input_values[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Predict Single Input"):
    try:
        input_scaled = preprocess_input(input_values, scalers)
        prediction = model.predict(input_scaled)[0, 0]
        st.write(f"### Probability of Default: {prediction:.2%}")
        st.write("### Prediction:", "Default" if prediction > 0.5 else "No Default")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Batch Prediction
st.header("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if not set(feature_names).issubset(data.columns):
            st.error("Uploaded file does not contain the correct features.")
        else:
            data = data[feature_names]
            data_scaled = scalers.transform(data)
            predictions = model.predict(data_scaled).flatten()
            data["Probability of Default"] = predictions
            data["Prediction"] = ["Default" if p > 0.5 else "No Default" for p in predictions]
            st.write(data)
            st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")
