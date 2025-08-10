import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- LOAD THE FINAL TRAINED MODEL ---
# We are using the final model that was trained on the 5 selected features.
try:
    model = joblib.load('ridge_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Make sure 'ridge_model.joblib' is in the same directory.")
    st.stop()

# --- DEFINE THE 5 FEATURES THE MODEL EXPECTS ---
# This list must exactly match the columns and order your final model was trained on.
final_feature_names = [
    'job_level_Middle', 
    'location_Asia-Pacific', 
    'location_Europe', 
    'location_Others', 
    'layoff_risk'
]

# --- SET UP THE USER INTERFACE ---
st.set_page_config(page_title="Employee Performance Prediction", layout="centered")
st.title("📈 Employee Performance Rating Predictor")
st.write("This app predicts an employee's performance rating. Please provide the details below based on the dataset.")

# --- CREATE INPUT WIDGETS FOR THE REQUIRED FEATURES ---
st.header("Employee Details")

# --- Create user-friendly inputs that match the dataset columns ---
job_level = st.selectbox(
    label="Job Level",
    options=['Junior', 'Middle', 'Senior'], # Options from the dataset
    index=1, 
    help="Select the employee's current job level (e.g., 'Middle')."
)

location = st.selectbox(
    label="Location",
    options=['India', 'Europe', 'Others', 'North America'], # Options from the dataset
    index=0, 
    help="Select the employee's work location. 'India' will be treated as 'Asia-Pacific'."
)

# This input now matches the categorical data in your CSV file.
layoff_risk_category = st.selectbox(
    label="Layoff Risk",
    options=['Low', 'Medium', 'High'], # Options from the dataset
    index=1,
    help="Select the layoff risk category for the employee."
)

# --- PREDICTION LOGIC ---
if st.button("Predict Performance Rating", type="primary"):
    
    # --- 1. Create a dictionary for the 5 features, initialized to 0 ---
    input_data = {col: 0 for col in final_feature_names}

    # --- 2. Convert user inputs into the numerical/one-hot format for the model ---
    
    # a) Map the categorical layoff_risk to a numerical value.
    # This assumes a mapping was done during model training (e.g., Low=25, Medium=50, High=75).
    risk_mapping = {'Low': 25, 'Medium': 50, 'High': 75}
    input_data['layoff_risk'] = risk_mapping[layoff_risk_category]
    
    # b) Check if the selected job_level corresponds to the 'job_level_Middle' feature
    if job_level == 'Middle':
        input_data['job_level_Middle'] = 1
    
    # c) Check which location feature needs to be activated.
    # We map 'India' from the dataset to the 'location_Asia-Pacific' feature.
    if location == 'India':
        input_data['location_Asia-Pacific'] = 1
    elif location == 'Europe':
        input_data['location_Europe'] = 1
    elif location == 'Others':
        input_data['location_Others'] = 1
        
    # --- 3. Create a DataFrame from the inputs ---
    # The DataFrame must have columns in the exact same order as the model's training data.
    input_df = pd.DataFrame([input_data], columns=final_feature_names)
    
    # --- 4. Make a prediction directly with the model ---
    prediction = model.predict(input_df)
    
    # --- DISPLAY THE RESULT ---
    st.subheader("Prediction Result")
    st.metric(label="Predicted Performance Rating", value=f"{prediction[0]:.2f}")

    with st.expander("Show Model Input"):
        st.write("The model made its prediction based on this final input format:")
        st.dataframe(input_df)
