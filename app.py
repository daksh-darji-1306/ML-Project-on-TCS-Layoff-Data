import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- LOAD THE SAVED MODELS AND DATA ---
# Use a try-except block to handle potential file not found errors
try:
    # Load the RFE object that knows which features to select
    rfe = joblib.load('rfe_selector.joblib')
    
    # Load the final trained Ridge model
    model = joblib.load('ridge_model.joblib')
    
    # IMPORTANT: Load the list of original column names that the RFE object was trained on.
    # You should save this list when you train your model.
    # For this example, we'll define it manually.
    # YOU MUST UPDATE THIS LIST to match your original training data columns in the correct order.
    original_columns = ['job_level_Junior', 'job_level_Middle', 'job_level_Senior',
       'department_AI/ML', 'department_Cloud Services',
       'department_Management', 'department_Others',
       'department_Software Development', 'department_Support',
       'department_Testing', 'location_Asia-Pacific', 'location_Europe',
       'location_India', 'location_North America', 'location_Others',
       'salary_band_High', 'salary_band_Low', 'salary_band_Medium', 'age',
       'years_of_experience', 'billable_days', 'bench_time_days',
       'training_hours', 'project_utilization', 'redeployment_attempts',
       'redeployment_success', 'layoff_risk']

except FileNotFoundError:
    st.error("Model or column data not found. Make sure 'rfe_selector.joblib', 'ridge_model.joblib', and your column list are correct.")
    st.stop() # Stop the app if models aren't loaded

# --- SET UP THE USER INTERFACE ---
st.set_page_config(page_title="Employee Performance Prediction", layout="centered")
st.title("📈 Employee Performance Rating Predictor")
st.write("This app predicts an employee's performance rating based on their profile. Please provide the details below.")

# --- CREATE INPUT WIDGETS FOR FEATURES ---
st.header("Employee Details")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    # --- CATEGORICAL INPUTS ---
    job_level = st.selectbox(
        label="Job Level",
        options=['Entry', 'Middle', 'Senior'],
        index=1 # Default to 'Middle'
    )
    
    location = st.selectbox(
        label="Location",
        options=['Asia-Pacific', 'Europe', 'North America', 'Others'],
        index=0 # Default to 'Asia-Pacific'
    )

    # --- NUMERICAL INPUTS ---
    layoff_risk = st.slider(
        label="Layoff Risk (%)",
        min_value=0,
        max_value=100,
        value=50, # Default value
        help="Estimated risk of the employee being laid off."
    )

with col2:
    # --- PLACEHOLDER INPUTS FOR OTHER FEATURES ---
    # Add other numerical inputs that your model was trained on.
    age = st.slider("Age", 20, 65, 35)
    satisfaction_score = st.slider("Satisfaction Score (1-10)", 1, 10, 7)
    projects_completed = st.number_input("Projects Completed", min_value=0, value=10)
    tenure_months = st.number_input("Tenure (Months)", min_value=0, value=24)


# --- PREDICTION LOGIC ---
if st.button("Predict Performance Rating", type="primary"):
    
    # --- 1. Create a dictionary to hold feature data ---
    # This makes it easier to build the input DataFrame
    input_data = {col: 0 for col in original_columns}

    # --- 2. Process inputs and perform one-hot encoding manually ---
    # Numerical features
    input_data['age'] = age
    input_data['satisfaction_score'] = satisfaction_score
    input_data['projects_completed'] = projects_completed
    input_data['tenure_months'] = tenure_months
    input_data['layoff_risk'] = layoff_risk

    # Manual one-hot encoding for 'job_level'
    if job_level in ['Entry', 'Middle', 'Senior']:
        input_data[f'job_level_{job_level}'] = 1
        
    # Manual one-hot encoding for 'location'
    if location in ['Asia-Pacific', 'Europe', 'North America', 'Others']:
        input_data[f'location_{location}'] = 1

    # --- 3. Create a DataFrame from the inputs ---
    # The DataFrame must have columns in the exact same order as the training data
    input_df = pd.DataFrame([input_data], columns=original_columns)
    
    # --- 4. Apply the RFE transformation ---
    # RFE will select the specific features it was trained on from the input_df
    input_rfe = rfe.transform(input_df)
    
    # --- 5. Make a prediction with the final model ---
    prediction = model.predict(input_rfe)
    
    # --- DISPLAY THE RESULT ---
    st.subheader("Prediction Result")
    st.metric(label="Predicted Performance Rating", value=f"{prediction[0]:.2f}")

    # Optional: Show the selected features that were used for the prediction
    with st.expander("Show features used for prediction"):
        selected_feature_names = [col for col, support in zip(original_columns, rfe.support_) if support]
        st.write("The prediction was based on these selected features:")
        st.json(selected_feature_names)
