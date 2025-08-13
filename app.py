import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- LOAD THE FINAL TRAINED MODEL & SCALER ---
try:
    model = joblib.load('ridge_model.joblib')
    scaler = joblib.load('scaler.joblib')  # You must save this from your training notebook
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

# --- DEFINE THE 5 FEATURES THE MODEL EXPECTS ---
final_feature_names = [
    'job_level_Middle', 
    'location_Asia-Pacific', 
    'location_Europe', 
    'location_Others', 
    'layoff_risk'
]

# --- STREAMLIT UI ---
st.set_page_config(page_title="Employee Performance Prediction", layout="centered")
st.title("📈 Employee Performance Rating Predictor")
st.write("Predict an employee's performance rating based on selected details.")

st.header("Employee Details")

job_level = st.selectbox(
    "Job Level",
    ['Junior', 'Middle', 'Senior'],
    index=1
)

location = st.selectbox(
    "Location",
    ['India', 'Europe', 'Others', 'North America'],
    index=0
)

layoff_risk_category = st.selectbox(
    "Layoff Risk",
    ['Low', 'Medium', 'High'],
    index=1
)

# --- PREDICTION ---
if st.button("Predict Performance Rating", type="primary"):

    # 1. Prepare feature dict
    input_data = {col: 0 for col in final_feature_names}

    # 2. Map layoff risk exactly as in training
    risk_mapping = {'Low': 25, 'Medium': 50, 'High': 75}
    input_data['layoff_risk'] = risk_mapping[layoff_risk_category]

    # 3. One-hot for job_level_Middle
    if job_level == 'Middle':
        input_data['job_level_Middle'] = 1

    # 4. One-hot for location
    if location == 'India':
        input_data['location_Asia-Pacific'] = 1
    elif location == 'Europe':
        input_data['location_Europe'] = 1
    elif location == 'Others':
        input_data['location_Others'] = 1

    # 5. Create DataFrame in correct column order
    input_df = pd.DataFrame([input_data], columns=final_feature_names)

    # 6. Apply SAME SCALER used in training
    input_scaled = scaler.transform(input_df)

    # 7. Predict & clip to 0–100 range
    prediction = model.predict(input_scaled)
    predicted_rating = np.clip(prediction[0], 0, 100)

    # 8. Show result
    st.subheader("Prediction Result")
    st.metric("Predicted Performance Rating", f"{predicted_rating:.2f}")

    with st.expander("Show Model Input"):
        st.write(input_df)
