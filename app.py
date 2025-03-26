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

# Streamlit app title
st.title('Credit Card Default Prediction')

# --- Home Page Information ---
st.header("Understanding the Input Features")
st.write("**SEX:**")
st.write("1: Male")
st.write("2: Female")
st.write("**EDUCATION:**")
st.write("1: Graduate School")
st.write("2: University")
st.write("3: High School")
st.write("**MARRIAGE:**")
st.write("1: Married")
st.write("2: Single")
st.write("3: Others")
st.write("**PAY_0 to PAY_6:** Payment status (-2 to 9, -2 indicating no consumption, -1 indicating paid in full, 0 indicating use of revolving credit, 1-9 indicating payment delay for 1-9 months)")

# Add a radio button to select prediction mode
prediction_mode = st.radio("Prediction Mode", ("Single Prediction", "Batch Prediction"))

# Define numerical_features outside the conditional blocks
numerical_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                      'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
                      'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

if prediction_mode == "Single Prediction":
    # Create input fields for features
    input_data = {}

    # Input fields with restrictions and dropdown menus
    input_data['LIMIT_BAL'] = st.number_input('LIMIT_BAL (Credit Limit)', min_value=0, step=1000)
    input_data['SEX'] = st.selectbox('SEX', [1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
    input_data['EDUCATION'] = st.selectbox('EDUCATION', [1, 2, 3], format_func=lambda x: 'Graduate School' if x == 1 else 'University' if x == 2 else 'High School')
    input_data['MARRIAGE'] = st.selectbox('MARRIAGE', [1, 2, 3], format_func=lambda x: 'Married' if x == 1 else 'Single' if x == 2 else 'Others')
    input_data['AGE'] = st.number_input('AGE', min_value=0, step=1)
    input_data['PAY_0'] = st.selectbox('PAY_0 (Repayment Status in September)', range(-2, 10))
    input_data['PAY_2'] = st.selectbox('PAY_2 (Repayment Status in August)', range(-2, 10))
    input_data['PAY_3'] = st.selectbox('PAY_3 (Repayment Status in July)', range(-2, 10))
    input_data['PAY_4'] = st.selectbox('PAY_4 (Repayment Status in June)', range(-2, 10))
    input_data['PAY_5'] = st.selectbox('PAY_5 (Repayment Status in May)', range(-2, 10))
    input_data['PAY_6'] = st.selectbox('PAY_6 (Repayment Status in April)', range(-2, 10))
    input_data['BILL_AMT1'] = st.number_input('BILL_AMT1 (Bill Statement Amount in September)', min_value=0, step=1000)
    input_data['BILL_AMT2'] = st.number_input('BILL_AMT2 (Bill Statement Amount in August)', min_value=0, step=1000)
    input_data['BILL_AMT3'] = st.number_input('BILL_AMT3 (Bill Statement Amount in July)', min_value=0, step=1000)
    input_data['BILL_AMT4'] = st.number_input('BILL_AMT4 (Bill Statement Amount in June)', min_value=0, step=1000)
    input_data['BILL_AMT5'] = st.number_input('BILL_AMT5 (Bill Statement Amount in May)', min_value=0, step=1000)
    input_data['BILL_AMT6'] = st.number_input('BILL_AMT6 (Bill Statement Amount in April)', min_value=0, step=1000)
    input_data['PAY_AMT1'] = st.number_input('PAY_AMT1 (Amount Paid in September)', min_value=0, step=1000)
    input_data['PAY_AMT2'] = st.number_input('PAY_AMT2 (Amount Paid in August)', min_value=0, step=1000)
    input_data['PAY_AMT3'] = st.number_input('PAY_AMT3 (Amount Paid in July)', min_value=0, step=1000)
    input_data['PAY_AMT4'] = st.number_input('PAY_AMT4 (Amount Paid in June)', min_value=0, step=1000)
    input_data['PAY_AMT5'] = st.number_input('PAY_AMT5 (Amount Paid in May)', min_value=0, step=1000)
    input_data['PAY_AMT6'] = st.number_input('PAY_AMT6 (Amount Paid in April)', min_value=0, step=1000)
    
    
    # Create a DataFrame from input data, making sure it has all expected columns
    input_df = pd.DataFrame([input_data], columns=feature_names)  # Use feature_names to ensure order and all columns
    
    # Create a Predict button
    if st.button("Predict"):
        # Standardize the input data using the first scaler
        input_df[numerical_features] = scalers[0].transform(input_df[numerical_features])
        
        # Make prediction
        prediction = model.predict(input_df)[0][0]

        # Display prediction
        st.subheader('Prediction')
        if prediction > 0.5:
            st.write('High risk of default (1)')
        else:
            st.write('Low risk of default (0)')

elif prediction_mode == "Batch Prediction":
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a Pandas DataFrame
        batch_df = pd.read_csv(uploaded_file)

        # Create a Predict button
        if st.button("Predict"):
            # Check if all required columns are present in the batch_df
            missing_cols = set(feature_names) - set(batch_df.columns)
            if missing_cols:
                st.error(f"Error: The following columns are missing in the uploaded CSV: {missing_cols}")
            else:
                # Standardize the input data using the first scaler
                batch_df[numerical_features] = scalers[0].transform(batch_df[numerical_features])

                # Make predictions
                predictions = model.predict(batch_df)

                # Add predictions to the DataFrame
                batch_df['Prediction'] = (predictions > 0.5).astype(int)  # Convert to 0 or 1

                # Display predictions
                st.subheader('Batch Predictions')
                st.write(batch_df)  # Display the DataFrame with predictions
