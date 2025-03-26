import streamlit as st
import pandas as pd
import pickle
from tensorflow import keras

# Load the trained global model
model = keras.models.load_model('federated_credit_card_model.keras')

# Load the scalers
with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

# Load the feature names
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load additional information
with open('additional_info.pkl', 'rb') as f:
    additional_info = pickle.load(f)

# Streamlit app title
st.title('Credit Card Default Prediction')

# Create input fields for features
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0)

# Create a DataFrame from input data
input_df = pd.DataFrame([input_data])

# Standardize the input data using the first scaler (assuming all scalers are similar)
numerical_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                      'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
                      'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
input_df[numerical_features] = scalers[0].transform(input_df[numerical_features])

# Make prediction
prediction = model.predict(input_df)[0][0]

# Display prediction
st.subheader('Prediction')
if prediction > 0.5:
    st.write('High risk of default')
else:
    st.write('Low risk of default')
